"""Placeholder worker process that occupies idle GPUs.

This module contains both the long-running placeholder ``main`` (spawned by
``gpulock _placeholder``) and the simple Unix-socket client helpers used by
``guard`` and lock-acquire paths to talk to a running placeholder worker.
"""

from __future__ import annotations

import contextlib
import logging
import os
import select
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Optional

from .gpu import pid_exists
from .paths import resolve_lock_root


# ---------------------------------------------------------------------------
# IPC client helpers
# ---------------------------------------------------------------------------

def placeholder_socket_path(gpu_dir: Path) -> Path:
    return gpu_dir / "placeholder.sock"


def placeholder_command(gpu_dir: Path, command: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    sock_path = placeholder_socket_path(gpu_dir)
    if not sock_path.exists():
        return (False, "missing socket")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_s)
            client.connect(str(sock_path))
            client.sendall((command.strip() + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except OSError as e:
        return (False, str(e))
    response = b"".join(chunks).decode("utf-8", errors="ignore").strip()
    if not response:
        return (False, "empty response")
    if response.startswith("ok"):
        return (True, response)
    return (False, response)


def park_placeholder(gpu_dir: Path, timeout_s: float = 5.0) -> bool:
    ok, _ = placeholder_command(gpu_dir, "park", timeout_s=timeout_s)
    return ok


def activate_placeholder(gpu_dir: Path, timeout_s: float = 5.0) -> bool:
    ok, _ = placeholder_command(gpu_dir, "activate", timeout_s=timeout_s)
    return ok


def stop_placeholder(gpu_dir: Path, timeout_s: float = 5.0) -> bool:
    ok, _ = placeholder_command(gpu_dir, "stop", timeout_s=timeout_s)
    return ok


def wait_placeholder_ready(gpu_dir: Path, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, _ = placeholder_command(gpu_dir, "status", timeout_s=0.5)
        if ok:
            return True
        time.sleep(0.05)
    return False


def placeholder_state(gpu_dir: Path, timeout_s: float = 1.0) -> Optional[str]:
    ok, response = placeholder_command(gpu_dir, "status", timeout_s=timeout_s)
    if not ok:
        return None
    for token in response.split():
        if token.startswith("state="):
            return token.split("=", 1)[1].strip()
    return None


def kill_placeholder(gpu_dir: Path) -> None:
    pid_file = gpu_dir / "placeholder.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    if pid_exists(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not pid_exists(pid):
                break
            time.sleep(0.1)
    pid_file.unlink(missing_ok=True)
    logging.getLogger("gpulock.guard").info(
        "%s: killed placeholder pid=%d (lock acquire)", gpu_dir.name, pid
    )


# ---------------------------------------------------------------------------
# CUDAGraph workload helpers (used only inside the placeholder process)
# ---------------------------------------------------------------------------

_PLACEHOLDER_WORKLOAD_NAME = "gentle-training"
_PLACEHOLDER_TARGET_REPLAY_S = 0.039
_PLACEHOLDER_IDLE_SLEEP_S = 0.008
_PLACEHOLDER_CALIBRATION_ITERS = 2
_PLACEHOLDER_REPLAY_TOLERANCE = 0.15
_PLACEHOLDER_SLEEP_CYCLES = 8_000_000
_PLACEHOLDER_FEATURE_ROWS = 1024
_PLACEHOLDER_FEATURE_COLS = 1024
_PLACEHOLDER_GEMM_DIM = 1024


def _round_up_multiple(value: int, granularity: int) -> int:
    granularity = max(granularity, 1)
    value = max(value, 1)
    return ((value + granularity - 1) // granularity) * granularity


def _cuda_elapsed_s(torch, stream, work) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.synchronize()
    with torch.cuda.stream(stream):
        start.record(stream)
        work()
        end.record(stream)
    end.synchronize()
    return max(start.elapsed_time(end) / 1000.0, 0.0)


def _make_placeholder_workload(torch) -> dict[str, object]:
    feature_shape = (_PLACEHOLDER_FEATURE_ROWS, _PLACEHOLDER_FEATURE_COLS)
    gemm_shape = (_PLACEHOLDER_GEMM_DIM, _PLACEHOLDER_GEMM_DIM)
    stats = torch.empty((_PLACEHOLDER_FEATURE_ROWS,), dtype=torch.float32, device="cuda:0")
    sleep_cycles = _PLACEHOLDER_SLEEP_CYCLES if hasattr(torch.cuda, "_sleep") else 0
    return {
        "feature": torch.randn(feature_shape, dtype=torch.float32, device="cuda:0"),
        "activation": torch.randn(feature_shape, dtype=torch.float32, device="cuda:0"),
        "gate": torch.randn(feature_shape, dtype=torch.float32, device="cuda:0"),
        "scratch": torch.empty(feature_shape, dtype=torch.float32, device="cuda:0"),
        "stats": stats,
        "stats_view": stats.view(_PLACEHOLDER_FEATURE_ROWS, 1),
        "inv_cols": 1.0 / float(_PLACEHOLDER_FEATURE_COLS),
        "sleep_cycles": sleep_cycles,
        "gemm_a": torch.randn(gemm_shape, dtype=torch.float16, device="cuda:0"),
        "gemm_b": torch.randn(gemm_shape, dtype=torch.float16, device="cuda:0"),
        "gemm_c": torch.empty(gemm_shape, dtype=torch.float16, device="cuda:0"),
    }


def _run_placeholder_workload_iteration(torch, workload: dict[str, object]) -> None:
    feature = workload["feature"]
    activation = workload["activation"]
    gate = workload["gate"]
    scratch = workload["scratch"]
    stats = workload["stats"]
    stats_view = workload["stats_view"]

    sleep_cycles = int(workload["sleep_cycles"])
    if sleep_cycles > 0:
        torch.cuda._sleep(sleep_cycles)
    torch.addcmul(feature, activation, gate, value=0.0007, out=scratch)
    torch.sigmoid(scratch, out=gate)
    torch.mul(scratch, gate, out=activation)
    torch.sum(activation, dim=1, out=stats)
    torch.mul(stats, workload["inv_cols"], out=stats)
    torch.add(activation, stats_view, alpha=-0.25, out=feature)
    torch.mm(workload["gemm_a"], workload["gemm_b"], out=workload["gemm_c"])
    torch.add(feature, scratch, alpha=0.125, out=feature)


def _measure_workload_loop_s(torch, stream, workload: dict[str, object], iters: int) -> float:
    def work():
        for _ in range(max(iters, 1)):
            _run_placeholder_workload_iteration(torch, workload)

    return _cuda_elapsed_s(torch, stream, work)


def _capture_placeholder_graph(torch, workload: dict[str, object], iters: int):
    capture_stream = torch.cuda.Stream()
    warmup_iters = min(max(iters, 1), 64)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(2):
            for _ in range(warmup_iters):
                _run_placeholder_workload_iteration(torch, workload)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(max(iters, 1)):
            _run_placeholder_workload_iteration(torch, workload)
    capture_stream.synchronize()
    return graph, capture_stream


def build_placeholder_graph(torch, workload: dict[str, object]):
    target_replay_s = _PLACEHOLDER_TARGET_REPLAY_S
    calibration_iters = _PLACEHOLDER_CALIBRATION_ITERS
    tolerance = _PLACEHOLDER_REPLAY_TOLERANCE
    calibration_stream = torch.cuda.Stream()
    eager_sample_s = _measure_workload_loop_s(torch, calibration_stream, workload, calibration_iters)
    if eager_sample_s <= 0.0:
        raise RuntimeError("failed to measure eager placeholder workload sample time")

    per_iter_s = eager_sample_s / calibration_iters
    graph_iters = _round_up_multiple(int(target_replay_s / per_iter_s), calibration_iters)
    graph_iters = max(graph_iters, calibration_iters)

    best_graph = None
    best_stream = None
    best_replay_s = 0.0
    for _ in range(2):
        graph, replay_stream = _capture_placeholder_graph(torch, workload, graph_iters)
        replay_s = _cuda_elapsed_s(torch, replay_stream, graph.replay)
        best_graph = graph
        best_stream = replay_stream
        best_replay_s = replay_s
        if replay_s <= 0.0:
            break
        if abs(replay_s - target_replay_s) / target_replay_s <= tolerance:
            break
        adjusted_iters = _round_up_multiple(int(graph_iters * target_replay_s / replay_s), calibration_iters)
        adjusted_iters = max(adjusted_iters, calibration_iters)
        if adjusted_iters == graph_iters:
            break
        graph_iters = adjusted_iters

    if best_graph is None or best_stream is None:
        raise RuntimeError("failed to build placeholder CUDAGraph")
    return best_graph, best_stream, graph_iters, eager_sample_s, best_replay_s


# ---------------------------------------------------------------------------
# Placeholder process entry point
# ---------------------------------------------------------------------------

def placeholder_main(gpu_id: int, mem_ratio: float = 0.85) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import setproctitle  # type: ignore
    import torch  # type: ignore

    setproctitle.setproctitle("tensorrt_engine_cache")

    lock_root = resolve_lock_root()
    gpu_dir = lock_root / f"gpu{gpu_id}"
    gpu_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pid_file = gpu_dir / "placeholder.pid"
    sock_path = placeholder_socket_path(gpu_dir)

    try:
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", None)
        if total is None:
            total = getattr(props, "total_mem", None)
        if total is None:
            _, total = torch.cuda.mem_get_info()
        total = int(total)
        count = int(total * max(0.0, min(1.0, mem_ratio))) // 4
        compute_only = mem_ratio <= 0.0
    except Exception as e:
        print(f"[gpulock] placeholder gpu{gpu_id}: {e}", file=sys.stderr)
        return 1

    idle_sleep_s = _PLACEHOLDER_IDLE_SLEEP_S
    state: dict[str, object] = {
        "buf": None,
        "graph": None,
        "replay_stream": None,
        "workload": None,
        "workload_iters": 0,
        "workload_sleep_cycles": 0,
        "eager_sample_s": 0.0,
        "replay_s": 0.0,
    }
    enabled = False
    stopping = False

    def release_resources() -> None:
        nonlocal enabled
        enabled = False
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        state["graph"] = None
        state["replay_stream"] = None
        state["workload"] = None
        state["buf"] = None
        state["workload_iters"] = 0
        state["workload_sleep_cycles"] = 0
        state["eager_sample_s"] = 0.0
        state["replay_s"] = 0.0
        with contextlib.suppress(Exception):
            import gc

            gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    def ensure_resources() -> None:
        if state["buf"] is None and not compute_only and count > 0:
            state["buf"] = torch.empty(count, dtype=torch.float32, device="cuda:0")
            print(
                f"[gpulock] placeholder gpu{gpu_id}: allocated {count * 4 / 1e9:.1f}GB",
                flush=True,
            )
        elif compute_only:
            print(
                f"[gpulock] placeholder gpu{gpu_id}: compute-only mode (no memory allocation)",
                flush=True,
            )
        if state["graph"] is not None:
            return
        workload = _make_placeholder_workload(torch)
        graph, replay_stream, workload_iters, eager_sample_s, replay_s = build_placeholder_graph(
            torch, workload
        )
        state["graph"] = graph
        state["replay_stream"] = replay_stream
        state["workload"] = workload
        state["workload_iters"] = workload_iters
        state["workload_sleep_cycles"] = int(workload["sleep_cycles"])
        state["eager_sample_s"] = eager_sample_s
        state["replay_s"] = replay_s
        mode_str = "compute-only" if compute_only else f"mem_ratio={mem_ratio:.2f}"
        active_fraction = replay_s / (replay_s + idle_sleep_s) if replay_s > 0.0 else 0.0
        print(
            f"[gpulock] placeholder gpu{gpu_id}: CUDAGraph {_PLACEHOLDER_WORKLOAD_NAME} workload enabled "
            f"({mode_str}, feature={_PLACEHOLDER_FEATURE_ROWS}x{_PLACEHOLDER_FEATURE_COLS} fp32, "
            f"tc_gemm={_PLACEHOLDER_GEMM_DIM}x{_PLACEHOLDER_GEMM_DIM} fp16, "
            f"sleep_cycles={int(workload['sleep_cycles'])}, "
            f"iters={workload_iters}, eager_sample={eager_sample_s:.6f}s/{_PLACEHOLDER_CALIBRATION_ITERS}, "
            f"replay={replay_s:.6f}s, sleep={idle_sleep_s:.3f}s, active_fraction={active_fraction:.2f})",
            flush=True,
        )

    def state_label() -> str:
        if enabled:
            return "active"
        return "parked"

    def handle_command(raw_command: str) -> str:
        nonlocal enabled, stopping
        command = raw_command.strip().lower() or "status"
        if command == "activate":
            ensure_resources()
            enabled = True
            return f"ok state={state_label()}"
        if command == "park":
            release_resources()
            return "ok state=parked"
        if command == "stop":
            release_resources()
            stopping = True
            return "ok state=stopping"
        if command == "status":
            return (
                f"ok state={state_label()} "
                f"workload={_PLACEHOLDER_WORKLOAD_NAME} "
                f"iters={int(state['workload_iters'])} replay_s={float(state['replay_s']):.6f} "
                f"sleep_s={idle_sleep_s:.6f} sleep_cycles={int(state['workload_sleep_cycles'])}"
            )
        return f"error unknown_command={command}"

    def accept_one(timeout_s: float) -> bool:
        ready, _, _ = select.select([server], [], [], max(timeout_s, 0.0))
        if not ready:
            return False
        try:
            conn, _ = server.accept()
        except OSError:
            return False
        with conn:
            data = b""
            while len(data) < 4096 and not data.endswith(b"\n"):
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                data += chunk
            response = handle_command(data.decode("utf-8", errors="ignore"))
            with contextlib.suppress(OSError):
                conn.sendall((response + "\n").encode("utf-8"))
        return True

    def cleanup() -> None:
        release_resources()
        with contextlib.suppress(Exception):
            server.close()
        with contextlib.suppress(Exception):
            if sock_path.exists():
                sock_path.unlink()
        with contextlib.suppress(Exception):
            if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink(missing_ok=True)

    def on_signal(_sig, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    sock_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    sock_path.chmod(0o600)
    server.listen(8)
    pid_file.write_text(str(os.getpid()))
    try:
        while not stopping:
            handled = False
            while accept_one(0.0):
                handled = True
                if stopping:
                    break
            if stopping:
                break
            if handled:
                continue
            if enabled:
                replay_stream = state["replay_stream"]
                graph = state["graph"]
                if replay_stream is None or graph is None:
                    release_resources()
                    continue
                with torch.cuda.stream(replay_stream):
                    graph.replay()
                replay_stream.synchronize()
                accept_one(idle_sleep_s)
                continue
            accept_one(1.0)
    finally:
        cleanup()
    return 0

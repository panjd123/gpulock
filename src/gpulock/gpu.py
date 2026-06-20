"""nvidia-smi helpers and GPU runtime probes."""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config import GpuRuntimeState


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return (False, "")
    return (result.returncode == 0, result.stdout)


def normalize_uuid(u: str) -> str:
    s = u.strip().lower()
    if s.startswith("gpu-"):
        s = s[4:]
    return s


def gpu_indices() -> list[int]:
    ok, out = run_cmd(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    if not ok or not out.strip():
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for line in out.splitlines():
        raw = line.split(",", 1)[0].strip()
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def gpu_uuid_by_index(index: int) -> Optional[str]:
    ok, out = run_cmd(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    if not ok or not out:
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx == index:
            return normalize_uuid(parts[1])
    return None


def gpu_has_processes_by_index(index: int) -> bool:
    target_uuid = gpu_uuid_by_index(index)
    if target_uuid is None:
        return True  # fail-safe: never auto-delete locks if nvidia-smi is unhappy
    ok, out = run_cmd(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    if not ok:
        return True
    if out.strip() == "":
        return False
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        if normalize_uuid(parts[0]) != target_uuid:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid > 0 and not is_placeholder_process(pid):
            return True
    return False


def process_owner_uid(pid: int) -> int | None:
    if pid <= 0:
        return None
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def gpu_compute_pids(index: int) -> set[int]:
    target_uuid = gpu_uuid_by_index(index)
    if target_uuid is None:
        return set()
    ok, out = run_cmd(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    if not ok or not out.strip():
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        if normalize_uuid(parts[0]) != target_uuid:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid > 0:
            pids.add(pid)
    return pids


def user_gpu_compute_pids(
    index: int,
    *,
    uid: int | None = None,
    exclude_pids: set[int] | None = None,
) -> set[int]:
    """Non-placeholder compute PIDs on ``index`` owned by ``uid`` (default: current user)."""
    owner_uid = os.getuid() if uid is None else uid
    excluded = exclude_pids or set()
    pids: set[int] = set()
    for pid in gpu_compute_pids(index):
        if pid in excluded or is_placeholder_process(pid):
            continue
        pid_uid = process_owner_uid(pid)
        if pid_uid == owner_uid:
            pids.add(pid)
    return pids


def is_placeholder_process(pid: int) -> bool:
    if pid <= 0:
        return False

    texts: list[str] = []
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    comm_path = Path(f"/proc/{pid}/comm")
    try:
        raw = cmdline_path.read_bytes()
        if raw:
            texts.append(raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").lower())
    except Exception:
        pass
    try:
        comm = comm_path.read_text(encoding="utf-8", errors="ignore").strip().lower()
        if comm:
            texts.append(comm)
    except Exception:
        pass
    if not texts:
        return False

    joined = " ".join(texts)
    if "tensorrt_engine_cache" in joined:
        return True
    if "_placeholder" in joined and "gpulock" in joined:
        return True
    return False


def gpu_runtime_state_by_index(index: int) -> Optional[GpuRuntimeState]:
    ok, out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not ok or not out.strip():
        return None

    util = None
    mem_used = None
    mem_total = None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx != index:
            continue
        try:
            util = int(float(parts[1]))
            mem_used = int(float(parts[2]))
            mem_total = int(float(parts[3]))
        except ValueError:
            return None
        break

    if util is None or mem_used is None or mem_total is None:
        return None

    pids = gpu_compute_pids(index)
    non_placeholder_pids = {pid for pid in pids if not is_placeholder_process(pid)}
    return GpuRuntimeState(
        util_gpu=max(util, 0),
        mem_used_mib=max(mem_used, 0),
        mem_total_mib=max(mem_total, 0),
        visible_compute_pids=len(pids),
        visible_non_placeholder_pids=len(non_placeholder_pids),
    )


def gpu_busy_reason_for_perf(index: int) -> tuple[bool, str]:
    state = gpu_runtime_state_by_index(index)
    if state is None:
        return (False, "runtime_state=unavailable")

    summary = (
        f"util={state.util_gpu}% mem={state.mem_used_mib}/{state.mem_total_mib}MiB "
        f"visible_compute_pids={state.visible_compute_pids} "
        f"visible_non_placeholder_pids={state.visible_non_placeholder_pids}"
    )
    # If the only compute processes are placeholders, the GPU is not considered
    # busy — the placeholder will be (or has been) parked before the lock is used.
    # Residual utilization from a recently-parked placeholder should not block
    # write lock acquisition.
    if state.visible_non_placeholder_pids == 0:
        return (False, f"{summary} idle_by=placeholder_only")
    if state.util_gpu > 0:
        return (True, f"{summary} busy_by=util={state.util_gpu}%")
    return (False, f"{summary} idle_by=util=0")


def kill_visible_placeholder_compute_pids(index: int) -> None:
    for pid in gpu_compute_pids(index):
        if not is_placeholder_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        # Wait up to 2s for SIGTERM to take effect.
        for _ in range(20):
            if not pid_exists(pid):
                break
            time.sleep(0.1)
        else:
            # SIGTERM didn't work; escalate to SIGKILL.
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            for _ in range(30):
                if not pid_exists(pid):
                    break
                time.sleep(0.1)


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        return e.errno == errno.EPERM
    return True

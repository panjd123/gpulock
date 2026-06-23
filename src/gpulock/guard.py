"""GPU guard daemon: enforces placeholders on idle GPUs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .activity import (
    init_guard_db,
    last_gpulock_activity_age_s,
    last_user_gpu_activity_age_s,
    has_recent_user_idle_activity,
    record_gpulock_activity,
    record_gpulock_event,
    record_user_gpu_activity,
)
from .gpu import (
    gpu_compute_pids,
    gpu_indices,
    gpu_runtime_state_by_index,
    kill_visible_placeholder_compute_pids,
    user_gpu_compute_pids,
)
from .lock import gpu_has_our_activity
from .logging_setup import setup_guard_logger
from .paths import read_lock_metadata, resolve_lock_root
from .placeholder import (
    activate_placeholder,
    kill_placeholder,
    park_placeholder,
    placeholder_command,
    placeholder_socket_path,
    placeholder_state,
    stop_placeholder,
)
from .service.common import (
    DEFAULT_GUARD_POLL_S,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PLACEHOLDER_IDLE_S,
    DEFAULT_PLACEHOLDER_MEM_RATIO,
    guard_status_path,
)


PLACEHOLDER_START_TIMEOUT_S = 60.0
PLACEHOLDER_START_FAILURE_EXIT_THRESHOLD = 3


def _open_guard_db(lock_root: Path) -> sqlite3.Connection:
    db_path = lock_root / "guard.db"
    conn = sqlite3.connect(str(db_path))
    init_guard_db(conn)
    conn.commit()
    return conn


# Backward-compatible alias for tests and callers.
_init_guard_db = _open_guard_db


def _has_recent_pulse(last_pulse_ts: dict[int, float], gpu_id: int, window_s: float) -> bool:
    ts = last_pulse_ts.get(gpu_id, 0.0)
    return ts > 0.0 and (time.time() - ts) <= max(window_s, 0.0)


def _resolve_guard_gpu_ids(requested_gpu_ids: list[int]) -> list[int]:
    if not requested_gpu_ids:
        return gpu_indices()
    deduped: list[int] = []
    seen: set[int] = set()
    for gpu_id in requested_gpu_ids:
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        deduped.append(gpu_id)
    return deduped


def _write_guard_status(lock_root: Path, snapshot: dict[str, object]) -> None:
    path = guard_status_path(lock_root)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)


def _external_compute_pids(visible_pids: set[int], known_placeholder_pids: set[int]) -> set[int]:
    return set(visible_pids) - set(known_placeholder_pids)


def _wait_placeholder_process_ready(
    gpu_dir: Path,
    proc: subprocess.Popen,
    timeout_s: float = PLACEHOLDER_START_TIMEOUT_S,
) -> bool:
    """Wait until the placeholder answers status, but fail fast if it exits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        ok, _ = placeholder_command(gpu_dir, "status", timeout_s=0.5)
        if ok:
            return True
        time.sleep(0.05)
    return False


def _collect_process_stderr(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    with contextlib.suppress(Exception):
        return proc.stderr.read().strip()
    return ""


def cmd_guard(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="gpulock guard")
    parser.add_argument(
        "gpu_ids", type=int, nargs="*", metavar="GPU_ID",
        help="GPU IDs to watch (default: all visible GPUs)",
    )
    parser.add_argument(
        "--idle-timeout", type=int, default=DEFAULT_IDLE_TIMEOUT,
        help=f"seconds without user activity before releasing placeholder (default {DEFAULT_IDLE_TIMEOUT})",
    )
    parser.add_argument(
        "--placeholder-idle-s", type=float, default=DEFAULT_PLACEHOLDER_IDLE_S,
        help=f"seconds of GPU idleness before spawning placeholder (default {DEFAULT_PLACEHOLDER_IDLE_S})",
    )
    parser.add_argument(
        "--guard-poll-s", type=float, default=DEFAULT_GUARD_POLL_S,
        help=f"guard main-loop poll interval in seconds (default {DEFAULT_GUARD_POLL_S})",
    )
    parser.add_argument(
        "--placeholder-mem-ratio", type=float, default=DEFAULT_PLACEHOLDER_MEM_RATIO,
        help=f"fraction of GPU memory to allocate (0.0-1.0, 0 = compute-only, default {DEFAULT_PLACEHOLDER_MEM_RATIO})",
    )
    args = parser.parse_args(argv)
    args.guard_poll_s = max(float(args.guard_poll_s), 0.05)
    args.gpu_ids = _resolve_guard_gpu_ids(args.gpu_ids)
    if not args.gpu_ids:
        print("[gpulock] could not enumerate visible GPUs for guard; pass GPU_ID explicitly", file=sys.stderr)
        return 1

    lock_root = resolve_lock_root()
    log = setup_guard_logger(lock_root)
    conn = _open_guard_db(lock_root)
    guard_uid = os.getuid()

    placeholders: dict[int, subprocess.Popen] = {}
    placeholder_fail_reported: set[int] = set()
    placeholder_started_at: dict[int, float] = {}
    placeholder_compute_pids_by_gpu: dict[int, set[int]] = {}
    placeholder_active: set[int] = set()
    consecutive_placeholder_start_failures = 0
    should_exit_for_placeholder_failures = False
    idle_since: dict[int, float] = {}
    dormant: set[int] = set()
    last_pulse_ts: dict[int, float] = {}
    def ingest_activity_pulse(gid: int) -> None:
        pulse_path = lock_root / f"gpu{gid}" / "activity.pulse"
        if not pulse_path.exists():
            return
        meta = read_lock_metadata(pulse_path)
        if not meta:
            return
        try:
            ts = float(meta.get("ts", "0"))
        except ValueError:
            return
        prev = last_pulse_ts.get(gid, 0.0)
        if ts <= prev:
            return
        last_pulse_ts[gid] = ts
        mode = meta.get("mode", "unknown")
        pid = meta.get("pid", "?")
        cmd = meta.get("cmdline", "").strip() or "<unknown>"
        record_gpulock_event(conn, gid, time.time())
        idle_since.pop(gid, None)
        if gid in dormant:
            dormant.discard(gid)
            log.info("gpu%d: woke from dormant (gpulock activity)", gid)
        log.info("gpu%d: gpulock activity mode=%s pid=%s cmd=%s", gid, mode, pid, cmd)

    def observe_user_gpu_activity(gid: int, now: float) -> set[int]:
        known_placeholder_pids = placeholder_compute_pids_by_gpu.get(gid, set())
        user_pids = user_gpu_compute_pids(
            gid,
            uid=guard_uid,
            exclude_pids=known_placeholder_pids,
        )
        if not user_pids:
            return user_pids
        record_user_gpu_activity(conn, gid, now)
        idle_since.pop(gid, None)
        if gid in dormant:
            dormant.discard(gid)
            log.info("gpu%d: woke from dormant (user GPU activity pids=%s)", gid, sorted(user_pids))
        return user_pids

    def ensure_placeholder_worker(gid: int) -> bool:
        nonlocal consecutive_placeholder_start_failures, should_exit_for_placeholder_failures
        gpu_dir = lock_root / f"gpu{gid}"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        existing = placeholders.get(gid)
        if existing is not None and existing.poll() is None:
            return True
        placeholder_active.discard(gid)
        placeholder_compute_pids_by_gpu.pop(gid, None)
        if existing is None:
            status_ok, _ = placeholder_command(gpu_dir, "status", timeout_s=0.5)
            if status_ok:
                stop_placeholder(gpu_dir, timeout_s=2.0)
                time.sleep(0.1)
            kill_placeholder(gpu_dir)
        placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
        compute_pids_before = gpu_compute_pids(gid)
        placeholder_cmd = [
            sys.executable, "-m", "gpulock", "_placeholder",
            str(gid),
            "--mem-ratio", str(args.placeholder_mem_ratio),
        ]
        proc = subprocess.Popen(
            placeholder_cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        placeholders[gid] = proc
        placeholder_started_at[gid] = time.time()
        if not _wait_placeholder_process_ready(gpu_dir, proc, timeout_s=PLACEHOLDER_START_TIMEOUT_S):
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            placeholders.pop(gid, None)
            placeholder_started_at.pop(gid, None)
            stderr_text = _collect_process_stderr(proc)
            if stderr_text:
                log.warning("gpu%d: placeholder worker failed to become ready: %s", gid, stderr_text)
            else:
                log.warning("gpu%d: placeholder worker failed to become ready", gid)
            consecutive_placeholder_start_failures += 1
            if consecutive_placeholder_start_failures >= PLACEHOLDER_START_FAILURE_EXIT_THRESHOLD:
                should_exit_for_placeholder_failures = True
                log.error(
                    "placeholder worker failed to start %d consecutive times; exiting guard for supervisor restart",
                    consecutive_placeholder_start_failures,
                )
            return False
        (gpu_dir / "placeholder.pid").write_text(str(proc.pid))
        compute_pids_after = gpu_compute_pids(gid)
        new_compute_pids = compute_pids_after - compute_pids_before
        if len(new_compute_pids) == 1:
            placeholder_compute_pids_by_gpu[gid] = set(new_compute_pids)
            log.info(
                "gpu%d: mapped placeholder compute pid(s) from nvidia-smi: %s",
                gid,
                sorted(new_compute_pids),
            )
        else:
            placeholder_compute_pids_by_gpu[gid] = set()
            log.warning(
                "gpu%d: could not uniquely map placeholder compute pid(s) "
                "(before=%s after=%s); treating visible compute pids as external until mapping is known",
                gid,
                sorted(compute_pids_before),
                sorted(compute_pids_after),
            )
        placeholder_fail_reported.discard(gid)
        consecutive_placeholder_start_failures = 0
        log.info("gpu%d: spawned placeholder worker (pid=%d)", gid, proc.pid)
        return True

    def park_placeholder_worker(gid: int, reason: str) -> None:
        gpu_dir = lock_root / f"gpu{gid}"
        if park_placeholder(gpu_dir, timeout_s=5.0):
            if gid in placeholder_active:
                log.info("gpu%d: parked placeholder (%s)", gid, reason)
            placeholder_active.discard(gid)
            return
        placeholder_active.discard(gid)
        placeholder_compute_pids_by_gpu.pop(gid, None)
        kill_placeholder(gpu_dir)
        kill_visible_placeholder_compute_pids(gid)
        log.warning("gpu%d: placeholder park failed, killed worker (%s)", gid, reason)

    def fully_stop_placeholder_worker(gid: int, reason: str) -> None:
        """Terminate the placeholder worker so its CUDA context is destroyed.

        Unlike park (which keeps the process and its resident CUDA context
        alive), this fully releases the GPU: the process exits and its context
        is gone. Used while a serve backend is starting up so the placeholder
        cannot interfere with startup-time autotuning. The guard will respawn a
        placeholder later via the normal idle/serve-idle paths.
        """
        gpu_dir = lock_root / f"gpu{gid}"
        proc = placeholders.get(gid)
        stopped = stop_placeholder(gpu_dir, timeout_s=5.0)
        if proc is not None:
            if proc.poll() is None and not stopped:
                with contextlib.suppress(Exception):
                    proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
        placeholders.pop(gid, None)
        placeholder_started_at.pop(gid, None)
        placeholder_compute_pids_by_gpu.pop(gid, None)
        placeholder_active.discard(gid)
        (gpu_dir / "placeholder.pid").unlink(missing_ok=True)
        placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
        log.info("gpu%d: fully stopped placeholder (%s)", gid, reason)

    def activate_placeholder_worker(gid: int, reason: str) -> None:
        if gid in placeholder_active:
            return
        gpu_dir = lock_root / f"gpu{gid}"
        if not ensure_placeholder_worker(gid):
            return
        if activate_placeholder(gpu_dir, timeout_s=10.0):
            placeholder_active.add(gid)
            log.info("gpu%d: activated placeholder (%s)", gid, reason)
            return
        log.warning("gpu%d: placeholder activate failed (%s)", gid, reason)

    def external_gpu_compute_pids(gid: int) -> set[int] | None:
        rt = gpu_runtime_state_by_index(gid)
        if rt is None:
            return None
        visible_pids = gpu_compute_pids(gid)
        known_placeholder_pids = placeholder_compute_pids_by_gpu.get(gid, set())
        return _external_compute_pids(visible_pids, known_placeholder_pids)

    def serve_signal_file(gid: int) -> Path:
        """Path to the serve-mode signal file for a GPU.

        When this file exists, placeholder must stay parked (the serve
        process has active requests). When it doesn't exist, placeholder
        may activate to keep utilization high.
        """
        return lock_root / f"gpu{gid}" / "serve.busy"

    def has_serve_signal(gid: int) -> bool:
        """Check if a serve-mode process has signaled it's busy."""
        return serve_signal_file(gid).exists()

    def serve_managed_file(gid: int) -> Path:
        """Path to the serve-managed marker file for a GPU.

        When this file exists, a ``gpulock serve`` reverse proxy owns the GPU.
        The proxy holds a write lock the whole time (so other gpulock jobs
        queue), which would normally make the guard park the placeholder
        unconditionally. The marker tells the guard to instead drive placeholder
        park/activate from the ``serve.busy`` signal: parked while requests are
        in flight, active (compute-only) when idle to keep utilization up.
        """
        return lock_root / f"gpu{gid}" / "serve.managed"

    def is_serve_managed(gid: int) -> bool:
        """Check if a serve reverse proxy currently owns this GPU."""
        return serve_managed_file(gid).exists()

    def serve_startup_file(gid: int) -> Path:
        """Path to the serve-startup marker file for a GPU.

        When this file exists, a ``gpulock serve`` backend on this GPU is still
        starting up (load/compile/warmup/autotune) and is not yet ready. The
        guard responds by fully **stopping** the placeholder (destroying its
        CUDA context), not just parking it, because a resident second context
        serializes the backend's startup-time autotuning and slows it
        several-fold. The placeholder is not respawned until the marker clears.
        """
        return lock_root / f"gpu{gid}" / "serve.startup"

    def is_serve_startup(gid: int) -> bool:
        """Check if a serve backend on this GPU is still starting up."""
        return serve_startup_file(gid).exists()

    def gpu_is_idle_for_placeholder(gid: int) -> bool:
        """Check if placeholder may activate on this GPU.

        GPU is idle for placeholder if:
        1. No gpulock lock/queue activity, AND
        2. No external compute PIDs, OR
           (external compute PIDs exist BUT serve signal is NOT present)

        The serve signal file is how gpulock serve processes tell the
        guard they have active requests and placeholder should stay parked.

        Exception: when the GPU is serve-managed, the serve proxy's own write
        lock is expected and must not block activation. In that case the
        decision is driven purely by the serve.busy signal (idle when absent).
        """
        if is_serve_managed(gid):
            return not has_serve_signal(gid)
        if gpu_has_our_activity(lock_root, gid):
            return False
        ext_pids = external_gpu_compute_pids(gid)
        if ext_pids is None:
            return True  # runtime unavailable, assume idle
        if not ext_pids:
            return True
        # External PIDs exist — check if serve signal says we should park
        return not has_serve_signal(gid)

    def gpu_status_snapshot(gid: int) -> dict[str, object]:
        gpu_dir = lock_root / f"gpu{gid}"
        proc = placeholders.get(gid)
        ph_alive = proc is not None and proc.poll() is None
        state = placeholder_state(gpu_dir, timeout_s=0.2) if ph_alive else None
        if state == "active":
            placeholder_status = "active"
        elif state == "parked":
            placeholder_status = "parked"
        elif ph_alive:
            placeholder_status = "unknown"
        else:
            placeholder_status = "not_running"

        now = time.time()
        our_active = gpu_has_our_activity(lock_root, gid)
        recent_pulse = _has_recent_pulse(last_pulse_ts, gid, max(args.placeholder_idle_s, 0.0))
        rt = gpu_runtime_state_by_index(gid)
        last_gpulock_activity_age = last_gpulock_activity_age_s(conn, gid, now)
        last_user_gpu_activity_age = last_user_gpu_activity_age_s(conn, gid, now)
        external_compute_pids = external_gpu_compute_pids(gid)
        user_gpu_pids = user_gpu_compute_pids(
            gid,
            uid=guard_uid,
            exclude_pids=placeholder_compute_pids_by_gpu.get(gid, set()),
        )

        reason = "placeholder active"
        serve_busy = has_serve_signal(gid)
        serve_managed = is_serve_managed(gid)
        serve_startup = is_serve_startup(gid)
        if serve_startup:
            reason = "released: serve backend starting up (placeholder fully stopped to free GPU for autotuning)"
        elif not ph_alive and gid in placeholder_fail_reported:
            reason = "placeholder worker failed to start; see guard log"
        elif gid in dormant:
            reason = f"dormant: no user GPU activity for {args.idle_timeout}s"
        elif serve_managed and serve_busy:
            reason = "parked: serve proxy has in-flight requests"
        elif serve_managed:
            reason = "active: serve proxy idle (no in-flight requests)"
        elif our_active:
            reason = "parked: gpulock lock or queue activity detected"
        elif rt is None:
            reason = "waiting: GPU runtime state unavailable from nvidia-smi"
        elif serve_busy:
            reason = "parked: serve signal file present (active requests)"
        elif user_gpu_pids:
            reason = (
                f"waiting: user-owned non-placeholder GPU process detected "
                f"(pids={sorted(user_gpu_pids)}, no serve signal → placeholder may activate)"
            )
        elif external_compute_pids:
            reason = (
                f"waiting: non-placeholder GPU process detected "
                f"(compute_pids={sorted(external_compute_pids)}, no serve signal → placeholder may activate)"
            )
        elif recent_pulse:
            reason = "waiting: recent gpulock activity pulse"
        elif gid in idle_since and placeholder_status != "active":
            wait_remaining = max(args.placeholder_idle_s, 0.0) - max(now - idle_since[gid], 0.0)
            reason = f"waiting: GPU idle grace period ({max(wait_remaining, 0.0):.1f}s remaining)"
        elif ph_alive and placeholder_status == "parked":
            reason = "parked: waiting for activation"
        elif not ph_alive:
            reason = "placeholder worker not running"

        runtime: dict[str, object] | None = None
        if rt is not None:
            runtime = {
                "util_gpu": rt.util_gpu,
                "mem_used_mib": rt.mem_used_mib,
                "mem_total_mib": rt.mem_total_mib,
                "visible_compute_pids": rt.visible_compute_pids,
                "visible_non_placeholder_pids": rt.visible_non_placeholder_pids,
                "known_placeholder_compute_pids": sorted(placeholder_compute_pids_by_gpu.get(gid, set())),
                "external_compute_pids": sorted(external_compute_pids or set()),
            }

        return {
            "gpu_id": gid,
            "placeholder": placeholder_status,
            "placeholder_pid": proc.pid if ph_alive and proc is not None else None,
            "dormant": gid in dormant,
            "our_activity": our_active,
            "serve_managed": serve_managed,
            "serve_startup": serve_startup,
            "serve_busy": serve_busy,
            "recent_pulse": recent_pulse,
            "idle_for_s": max(now - idle_since[gid], 0.0) if gid in idle_since else None,
            "last_gpulock_activity_age_s": last_gpulock_activity_age,
            "last_user_gpu_activity_age_s": last_user_gpu_activity_age,
            "user_gpu_compute_pids": sorted(user_gpu_pids),
            "runtime": runtime,
            "reason": reason,
        }

    def write_status_snapshot() -> None:
        now = time.time()
        _write_guard_status(
            lock_root,
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "updated_at": now,
                "updated_at_text": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(now)),
                "gpu_ids": args.gpu_ids,
                "idle_timeout": args.idle_timeout,
                "placeholder_idle_s": args.placeholder_idle_s,
                "guard_poll_s": args.guard_poll_s,
                "placeholder_mem_ratio": args.placeholder_mem_ratio,
                "gpus": [gpu_status_snapshot(gid) for gid in args.gpu_ids],
            },
        )

    def cleanup() -> None:
        for gid, proc in placeholders.items():
            gpu_dir = lock_root / f"gpu{gid}"
            if proc.poll() is None and not stop_placeholder(gpu_dir, timeout_s=5.0):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            (gpu_dir / "placeholder.pid").unlink(missing_ok=True)
            placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
        placeholders.clear()
        placeholder_started_at.clear()
        placeholder_compute_pids_by_gpu.clear()
        placeholder_active.clear()
        guard_status_path(lock_root).unlink(missing_ok=True)
        conn.close()

    def on_signal(_sig, _frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Seed activity rows so a fresh guard start does not immediately go dormant.
    startup_ts = time.time()
    for gid in args.gpu_ids:
        record_gpulock_activity(conn, gid, startup_ts)
        observe_user_gpu_activity(gid, startup_ts)
        # Do not spawn a placeholder on a GPU whose serve backend is still
        # starting up: it must stay free of any second CUDA context.
        if is_serve_startup(gid):
            continue
        ensure_placeholder_worker(gid)
        if should_exit_for_placeholder_failures:
            cleanup()
            return 70
    conn.commit()

    log.info(
        "watching gpu %s (placeholder_idle_s=%.3f guard_poll_s=%.3f)",
        args.gpu_ids,
        max(args.placeholder_idle_s, 0.0),
        args.guard_poll_s,
    )
    for gid in args.gpu_ids:
        if gpu_has_our_activity(lock_root, gid):
            continue
        if is_serve_startup(gid):
            continue
        # Activate placeholder if GPU is idle for placeholder at startup.
        if gpu_is_idle_for_placeholder(gid):
            activate_placeholder_worker(gid, "guard startup idle")
            if should_exit_for_placeholder_failures:
                cleanup()
                return 70
    write_status_snapshot()

    try:
        while True:
            for gid in args.gpu_ids:
                gpu_dir = lock_root / f"gpu{gid}"
                gpu_dir.mkdir(parents=True, exist_ok=True)
                pid_file = gpu_dir / "placeholder.pid"

                ingest_activity_pulse(gid)
                now = time.time()
                user_gpu_pids = observe_user_gpu_activity(gid, now)

                if gid in placeholders and placeholders[gid].poll() is not None:
                    rc = placeholders[gid].returncode
                    stderr_text = ""
                    if placeholders[gid].stderr is not None:
                        with contextlib.suppress(Exception):
                            stderr_text = placeholders[gid].stderr.read().strip()
                    del placeholders[gid]
                    placeholder_started_at.pop(gid, None)
                    placeholder_compute_pids_by_gpu.pop(gid, None)
                    placeholder_active.discard(gid)
                    pid_file.unlink(missing_ok=True)
                    placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
                    if rc is None:
                        rc = -1
                    if rc != 0 and gid not in placeholder_fail_reported:
                        placeholder_fail_reported.add(gid)
                        if stderr_text:
                            log.warning(
                                "gpu%d: placeholder start failed (rc=%d): %s; suppressing repeated errors until it runs successfully",
                                gid, rc, stderr_text,
                            )
                        else:
                            log.warning(
                                "gpu%d: placeholder start failed (rc=%d), suppressing repeated errors until it runs successfully",
                                gid, rc,
                            )

                ph_alive = gid in placeholders and placeholders[gid].poll() is None
                if ph_alive and gid in placeholder_fail_reported:
                    started = placeholder_started_at.get(gid, 0.0)
                    if time.time() - started >= 10.0:
                        placeholder_fail_reported.discard(gid)
                        log.info("gpu%d: placeholder running stably, cleared startup-failure suppression", gid)
                if ph_alive:
                    state = placeholder_state(gpu_dir, timeout_s=0.5)
                    if state == "active":
                        placeholder_active.add(gid)
                    elif state == "parked":
                        placeholder_active.discard(gid)
                else:
                    placeholder_active.discard(gid)

                our_active = gpu_has_our_activity(lock_root, gid)
                recent_pulse = _has_recent_pulse(last_pulse_ts, gid, max(args.placeholder_idle_s, 0.0))
                serve_busy = has_serve_signal(gid)
                serve_managed = is_serve_managed(gid)

                if is_serve_startup(gid):
                    # Serve backend is still starting up: fully stop the
                    # placeholder (destroy its CUDA context) so it cannot
                    # interfere with startup-time autotuning, and do not respawn
                    # until the backend is ready (marker cleared).
                    if gid in placeholders or ph_alive:
                        fully_stop_placeholder_worker(gid, "serve backend starting up")
                    idle_since.pop(gid, None)
                elif our_active and not serve_managed:
                    record_gpulock_activity(conn, gid, time.time())
                    if ph_alive:
                        park_placeholder_worker(gid, "our process/lock detected")
                    idle_since.pop(gid, None)
                    if gid in dormant:
                        dormant.discard(gid)
                        log.info("gpu%d: woke from dormant (our activity detected)", gid)
                elif serve_busy:
                    # Serve process has active requests — keep placeholder parked.
                    # (Covers both serve-managed GPUs and the legacy hook flow.)
                    if ph_alive and gid in placeholder_active:
                        park_placeholder_worker(gid, "serve signal present")
                    idle_since.pop(gid, None)
                    if gid in dormant:
                        dormant.discard(gid)
                elif serve_managed:
                    # Serve proxy owns the GPU and has no in-flight requests:
                    # activate the placeholder (compute-only) to keep utilization
                    # up. Its own write lock is expected and must not block this.
                    if not (ph_alive and gid in placeholder_active):
                        activate_placeholder_worker(gid, "serve idle (no requests)")
                        if should_exit_for_placeholder_failures:
                            cleanup()
                            return 70
                    idle_since.pop(gid, None)
                    if gid in dormant:
                        dormant.discard(gid)
                elif gid in dormant:
                    if ph_alive and gid in placeholder_active:
                        park_placeholder_worker(gid, "dormant")
                else:
                    if gid not in idle_since:
                        idle_since[gid] = now
                    if recent_pulse or user_gpu_pids:
                        idle_since[gid] = now
                    elif now - idle_since[gid] >= max(args.placeholder_idle_s, 0.0):
                        # Activate placeholder if GPU is idle for it:
                        # no lock activity, and either no external PIDs
                        # or external PIDs exist but no serve signal.
                        if gpu_is_idle_for_placeholder(gid):
                            activate_placeholder_worker(gid, "gpu idle")
                            if should_exit_for_placeholder_failures:
                                cleanup()
                                return 70
                            if gid in placeholder_active:
                                idle_since.pop(gid, None)
                        else:
                            idle_since[gid] = now  # reset idle timer
                    if (
                        ph_alive
                        and gid in placeholder_active
                        and not has_recent_user_idle_activity(conn, gid, args.idle_timeout, now=now)
                    ):
                        park_placeholder_worker(gid, f"no user activity for {args.idle_timeout}s")
                        dormant.add(gid)
                        idle_since.pop(gid, None)
                        log.info("gpu%d: dormant (no user activity for %ds)", gid, args.idle_timeout)

            conn.commit()
            write_status_snapshot()
            time.sleep(args.guard_poll_s)
    finally:
        cleanup()

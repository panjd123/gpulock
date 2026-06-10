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

from .gpu import gpu_compute_pids, gpu_indices, gpu_runtime_state_by_index, kill_visible_placeholder_compute_pids
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
from .service.common import DEFAULT_IDLE_TIMEOUT, DEFAULT_PLACEHOLDER_IDLE_S
from .service.common import guard_status_path


PLACEHOLDER_START_TIMEOUT_S = 60.0
PLACEHOLDER_START_FAILURE_EXIT_THRESHOLD = 3


def _init_guard_db(lock_root: Path) -> sqlite3.Connection:
    db_path = lock_root / "guard.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpu_activity (ts REAL NOT NULL, gpu_id INTEGER NOT NULL, active INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_activity(gpu_id, ts)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpu_last_activity (gpu_id INTEGER PRIMARY KEY, last_activity_ts REAL NOT NULL)"
    )
    conn.commit()
    return conn


def _touch_last_activity(conn: sqlite3.Connection, gpu_id: int, ts: float) -> None:
    conn.execute(
        "INSERT INTO gpu_last_activity(gpu_id, last_activity_ts) VALUES (?, ?) "
        "ON CONFLICT(gpu_id) DO UPDATE SET last_activity_ts=excluded.last_activity_ts",
        (gpu_id, ts),
    )


def _record_activity_event(conn: sqlite3.Connection, gpu_id: int, ts: float) -> None:
    conn.execute("INSERT INTO gpu_activity VALUES (?,?,?)", (ts, gpu_id, 1))
    _touch_last_activity(conn, gpu_id, ts)


def _has_recent_activity(
    conn: sqlite3.Connection,
    gpu_id: int,
    window_s: float = DEFAULT_IDLE_TIMEOUT,
) -> bool:
    row = conn.execute(
        "SELECT last_activity_ts FROM gpu_last_activity WHERE gpu_id=?",
        (gpu_id,),
    ).fetchone()
    if row is None:
        return False
    try:
        last_ts = float(row[0])
    except (TypeError, ValueError):
        return False
    return (time.time() - last_ts) <= max(window_s, 0.0)


def _prune_activity_history(conn: sqlite3.Connection, now_ts: float, retention_s: float = 86400.0) -> int:
    cutoff = now_ts - max(retention_s, 0.0)
    cur = conn.execute("DELETE FROM gpu_activity WHERE ts<?", (cutoff,))
    return int(cur.rowcount or 0)


def _has_recent_pulse(last_pulse_ts: dict[int, float], gpu_id: int, window_s: float) -> bool:
    ts = last_pulse_ts.get(gpu_id, 0.0)
    return ts > 0.0 and (time.time() - ts) <= max(window_s, 0.0)


def _guard_poll_interval_s(placeholder_idle_s: float) -> float:
    return min(max(float(placeholder_idle_s) / 2.0, 0.05), 0.5)


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


def _last_activity_age_s(conn: sqlite3.Connection, gpu_id: int, now: float) -> float | None:
    row = conn.execute(
        "SELECT last_activity_ts FROM gpu_last_activity WHERE gpu_id=?",
        (gpu_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        return max(now - float(row[0]), 0.0)
    except (TypeError, ValueError):
        return None


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
    args = parser.parse_args(argv)
    args.gpu_ids = _resolve_guard_gpu_ids(args.gpu_ids)
    if not args.gpu_ids:
        print("[gpulock] could not enumerate visible GPUs for guard; pass GPU_ID explicitly", file=sys.stderr)
        return 1

    lock_root = resolve_lock_root()
    log = setup_guard_logger(lock_root)
    conn = _init_guard_db(lock_root)

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
    last_history_prune_ts = 0.0

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
        _record_activity_event(conn, gid, time.time())
        idle_since.pop(gid, None)
        if gid in dormant:
            dormant.discard(gid)
            log.info("gpu%d: woke from dormant (gpulock activity)", gid)
        log.info("gpu%d: gpulock activity mode=%s pid=%s cmd=%s", gid, mode, pid, cmd)

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
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "gpulock", "_placeholder",
                str(gid),
            ],
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
        last_activity_age = _last_activity_age_s(conn, gid, now)
        external_compute_pids = external_gpu_compute_pids(gid)

        reason = "placeholder active"
        if not ph_alive and gid in placeholder_fail_reported:
            reason = "placeholder worker failed to start; see guard log"
        elif gid in dormant:
            reason = f"dormant: no gpulock activity for {args.idle_timeout}s"
        elif our_active:
            reason = "parked: gpulock lock or queue activity detected"
        elif rt is None:
            reason = "waiting: GPU runtime state unavailable from nvidia-smi"
        elif external_compute_pids:
            reason = (
                "waiting: non-placeholder GPU process detected "
                f"(compute_pids={sorted(external_compute_pids)})"
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
            "recent_pulse": recent_pulse,
            "idle_for_s": max(now - idle_since[gid], 0.0) if gid in idle_since else None,
            "last_activity_age_s": last_activity_age,
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

    # Seed last-activity timestamps so a fresh guard start does not
    # immediately classify GPUs as dormant before any user job arrives.
    startup_ts = time.time()
    for gid in args.gpu_ids:
        _touch_last_activity(conn, gid, startup_ts)
        ensure_placeholder_worker(gid)
        if should_exit_for_placeholder_failures:
            cleanup()
            return 70
    conn.commit()
    _prune_activity_history(conn, startup_ts)
    conn.commit()

    log.info(
        "watching gpu %s (placeholder_idle_s=%.3f)",
        args.gpu_ids,
        max(args.placeholder_idle_s, 0.0),
    )
    for gid in args.gpu_ids:
        if gpu_has_our_activity(lock_root, gid):
            continue
        # Don't activate placeholder if non-placeholder processes are using GPU.
        rt = gpu_runtime_state_by_index(gid)
        external_compute_pids = external_gpu_compute_pids(gid)
        if external_compute_pids:
            continue
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

                if our_active:
                    _touch_last_activity(conn, gid, time.time())
                    if ph_alive:
                        park_placeholder_worker(gid, "our process/lock detected")
                    idle_since.pop(gid, None)
                    if gid in dormant:
                        dormant.discard(gid)
                        log.info("gpu%d: woke from dormant (our activity detected)", gid)
                elif gid in dormant:
                    if ph_alive and gid in placeholder_active:
                        park_placeholder_worker(gid, "dormant")
                else:
                    now = time.time()
                    if gid not in idle_since:
                        idle_since[gid] = now
                    if recent_pulse:
                        idle_since[gid] = now
                    elif now - idle_since[gid] >= max(args.placeholder_idle_s, 0.0):
                        # Do not reactivate placeholder if non-placeholder processes
                        # are actively using this GPU (e.g. training outlived its
                        # gpulock wrapper).
                        rt = gpu_runtime_state_by_index(gid)
                        external_compute_pids = external_gpu_compute_pids(gid)
                        if external_compute_pids:
                            idle_since[gid] = now  # reset idle timer
                        else:
                            activate_placeholder_worker(gid, "gpu idle")
                            if should_exit_for_placeholder_failures:
                                cleanup()
                                return 70
                            if gid in placeholder_active:
                                idle_since.pop(gid, None)
                    if ph_alive and gid in placeholder_active and not _has_recent_activity(conn, gid, args.idle_timeout):
                        park_placeholder_worker(gid, f"no user activity for {args.idle_timeout}s")
                        dormant.add(gid)
                        idle_since.pop(gid, None)
                        log.info("gpu%d: dormant (no user activity for %ds)", gid, args.idle_timeout)

            conn.commit()
            write_status_snapshot()
            now_ts = time.time()
            if now_ts - last_history_prune_ts >= 86400.0:
                deleted = _prune_activity_history(conn, now_ts)
                conn.commit()
                last_history_prune_ts = now_ts
                if deleted > 0:
                    log.info("pruned %d gpu_activity rows older than 24h", deleted)
            time.sleep(_guard_poll_interval_s(args.placeholder_idle_s))
    finally:
        cleanup()

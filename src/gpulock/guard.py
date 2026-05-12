"""GPU guard daemon: enforces placeholders on idle GPUs."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .gpu import gpu_indices, kill_visible_placeholder_compute_pids
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
    wait_placeholder_ready,
)


def _init_guard_db(lock_root: Path) -> sqlite3.Connection:
    db_path = lock_root / "guard.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpu_activity (ts REAL NOT NULL, gpu_id INTEGER NOT NULL, active INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_activity(gpu_id, ts)")
    conn.commit()
    return conn


def _has_recent_activity(conn: sqlite3.Connection, gpu_id: int, window_s: float = 5400) -> bool:
    cutoff = time.time() - window_s
    row = conn.execute(
        "SELECT COUNT(*) FROM gpu_activity WHERE gpu_id=? AND active=1 AND ts>?",
        (gpu_id, cutoff),
    ).fetchone()
    return row[0] > 0


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


def cmd_guard(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="gpulock guard")
    parser.add_argument(
        "gpu_ids", type=int, nargs="*", metavar="GPU_ID",
        help="GPU IDs to watch (default: all visible GPUs)",
    )
    parser.add_argument(
        "--idle-timeout", type=int, default=5400,
        help="seconds without user activity before releasing placeholder (default 5400)",
    )
    parser.add_argument(
        "--placeholder-idle-s", type=float, default=0.0,
        help="seconds of GPU idleness before spawning placeholder (default 0.0)",
    )
    parser.add_argument(
        "--placeholder-load", dest="placeholder_load", action="store_true", default=True,
        help="keep a compute loop in placeholder so GPU util stays non-zero (default: enabled)",
    )
    parser.add_argument(
        "--no-placeholder-load", dest="placeholder_load", action="store_false",
        help="disable placeholder compute loop and only reserve memory",
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
    placeholder_active: set[int] = set()
    idle_since: dict[int, float] = {}
    dormant: set[int] = set()
    last_pulse_ts: dict[int, float] = {}
    clean_counter = 0

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
        conn.execute("INSERT INTO gpu_activity VALUES (?,?,?)", (time.time(), gid, 1))
        idle_since.pop(gid, None)
        if gid in dormant:
            dormant.discard(gid)
            log.info("gpu%d: woke from dormant (gpulock activity)", gid)
        log.info("gpu%d: gpulock activity mode=%s pid=%s cmd=%s", gid, mode, pid, cmd)

    def ensure_placeholder_worker(gid: int) -> bool:
        gpu_dir = lock_root / f"gpu{gid}"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        existing = placeholders.get(gid)
        if existing is not None and existing.poll() is None:
            return True
        placeholder_active.discard(gid)
        if existing is None:
            status_ok, _ = placeholder_command(gpu_dir, "status", timeout_s=0.5)
            if status_ok:
                stop_placeholder(gpu_dir, timeout_s=2.0)
                time.sleep(0.1)
            kill_placeholder(gpu_dir)
        placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "gpulock", "_placeholder",
                str(gid), "1" if args.placeholder_load else "0",
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        placeholders[gid] = proc
        placeholder_started_at[gid] = time.time()
        if not wait_placeholder_ready(gpu_dir, timeout_s=60.0):
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            placeholders.pop(gid, None)
            placeholder_started_at.pop(gid, None)
            stderr_text = ""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr_text = proc.stderr.read().strip()
            if stderr_text:
                log.warning("gpu%d: placeholder worker failed to become ready: %s", gid, stderr_text)
            else:
                log.warning("gpu%d: placeholder worker failed to become ready", gid)
            return False
        (gpu_dir / "placeholder.pid").write_text(str(proc.pid))
        placeholder_fail_reported.discard(gid)
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
        placeholder_active.clear()
        conn.close()

    def on_signal(_sig, _frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Treat guard startup as one user activity pulse so placeholder is not
    # immediately considered idle/dormant before user starts real workloads.
    startup_ts = time.time()
    for gid in args.gpu_ids:
        conn.execute("INSERT INTO gpu_activity VALUES (?,?,?)", (startup_ts, gid, 1))
        ensure_placeholder_worker(gid)
    conn.commit()

    log.info(
        "watching gpu %s (placeholder_load=%s, placeholder_idle_s=%.3f)",
        args.gpu_ids,
        "on" if args.placeholder_load else "off",
        max(args.placeholder_idle_s, 0.0),
    )
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
                    if state in ("active", "reserved"):
                        placeholder_active.add(gid)
                    elif state == "parked":
                        placeholder_active.discard(gid)
                else:
                    placeholder_active.discard(gid)

                our_active = gpu_has_our_activity(lock_root, gid)
                recent_pulse = _has_recent_pulse(last_pulse_ts, gid, 3.0)

                conn.execute("INSERT INTO gpu_activity VALUES (?,?,?)", (time.time(), gid, int(our_active)))

                if our_active:
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
                        activate_placeholder_worker(gid, "gpu idle")
                        if gid in placeholder_active:
                            idle_since.pop(gid, None)
                    if ph_alive and gid in placeholder_active and not _has_recent_activity(conn, gid, args.idle_timeout):
                        park_placeholder_worker(gid, f"no user activity for {args.idle_timeout}s")
                        dormant.add(gid)
                        idle_since.pop(gid, None)
                        log.info("gpu%d: dormant (no user activity for %ds)", gid, args.idle_timeout)

            conn.commit()
            clean_counter += 1
            if clean_counter >= 3600:  # prune old records every ~1 hour
                conn.execute("DELETE FROM gpu_activity WHERE ts<?", (time.time() - 7200,))
                conn.commit()
                clean_counter = 0
            time.sleep(1)
    finally:
        cleanup()

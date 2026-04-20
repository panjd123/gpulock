"""GpuBenchLock: per-GPU read/write lock with heartbeats and orphan cleanup."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import logging
import os
import shlex
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import LockConfig, ProbeState, READ_MODE, WRITE_MODE
from .gpu import (
    gpu_busy_reason_for_perf,
    gpu_has_processes_by_index,
    kill_visible_placeholder_compute_pids,
    pid_exists,
)
from .paths import (
    read_last_heartbeat_ms,
    read_lock_metadata,
    read_lock_pid,
    resolve_lock_root,
)
from .placeholder import kill_placeholder, park_placeholder


def notify_guard_activity(lock_root: Path, gpu_id: int, mode: str) -> None:
    gpu_dir = lock_root / f"gpu{gpu_id}"
    try:
        gpu_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except Exception:
        return

    now = time.time()
    cmdline = " ".join(shlex.quote(x) for x in sys.argv)
    pulse_path = gpu_dir / "activity.pulse"
    tmp_path = gpu_dir / f".activity.pulse.{os.getpid()}"
    payload = (
        f"ts={now:.6f}\n"
        f"pid={os.getpid()}\n"
        f"mode={mode}\n"
        f"cmdline={cmdline}\n"
    )

    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, pulse_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)

    # Persist one activity pulse to DB so short-lived jobs are not missed.
    import sqlite3

    db_path = lock_root / "guard.db"
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS gpu_activity (ts REAL NOT NULL, gpu_id INTEGER NOT NULL, active INTEGER NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_activity(gpu_id, ts)")
        conn.execute("INSERT INTO gpu_activity VALUES (?,?,?)", (now, gpu_id, 1))
        conn.commit()
        conn.close()
    except Exception:
        pass


def gpu_has_our_activity(lock_root: Path, gpu_id: int) -> bool:
    gpu_dir = lock_root / f"gpu{gpu_id}"
    candidates: list[Path] = []

    wl = gpu_dir / "write.lock"
    if wl.exists():
        candidates.append(wl)

    rd = gpu_dir / "readers"
    if rd.is_dir():
        candidates.extend(rd.glob("*.lock"))

    qd = gpu_dir / "queue"
    if qd.is_dir():
        candidates.extend(qd.glob("*.req"))

    for path in candidates:
        pid = read_lock_pid(path)
        if pid is None:
            continue
        if pid_exists(pid):
            return True
    return False


class GpuBenchLock:
    def __init__(
        self,
        physical_device_id: int,
        mode: str,
        config: Optional[LockConfig] = None,
        wait_gpu_idle: bool = False,
        idle_streak_s: int = 3,
        idle_check_ms: int = 100,
    ):
        if mode not in (READ_MODE, WRITE_MODE):
            raise ValueError(f"Unsupported mode: {mode}")

        self.physical_device_id = int(physical_device_id)
        self.mode = mode
        self.config = config or LockConfig()
        self.root = resolve_lock_root()

        self.gpu_dir = self.root / f"gpu{self.physical_device_id}"
        self.readers_dir = self.gpu_dir / "readers"
        self.queue_dir = self.gpu_dir / "queue"
        self.queue_seq_path = self.gpu_dir / "queue.seq"
        self.writer_path = self.gpu_dir / "write.lock"
        self.gate_path = self.gpu_dir / "state.lock"

        self.lock_path: Optional[Path] = None
        self.fd: Optional[int] = None
        self.start_time = time.time()
        self.wait_gpu_idle = bool(wait_gpu_idle)
        self.idle_streak_s = max(int(idle_streak_s), 1)
        self.idle_check_ms = max(int(idle_check_ms), 100)

        self._stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._old_handlers: dict[int, signal.Handlers] = {}  # type: ignore[name-defined]
        self._registered_atexit = False
        self._orphan_probe: dict[str, ProbeState] = {}
        self._queue_request_path: Optional[Path] = None
        self._queue_request_seq: Optional[int] = None

        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.gpu_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.readers_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.queue_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _state_gate(self):
        gate_fd = os.open(self.gate_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(gate_fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(gate_fd, fcntl.LOCK_UN)
            finally:
                os.close(gate_fd)

    def _lock_payload(self, now_ms: int) -> str:
        cmdline = " ".join(shlex.quote(x) for x in sys.argv)
        return (
            f"pid={os.getpid()}\n"
            f"hostname={socket.gethostname()}\n"
            f"device_id={self.physical_device_id}\n"
            f"lock_mode={self.mode}\n"
            f"start_time={self.start_time:.3f}\n"
            f"cmdline={cmdline}\n"
            f"last_heartbeat_ms={now_ms}\n"
        )

    def _write_heartbeat(self) -> None:
        if self.fd is None:
            return
        payload = self._lock_payload(int(time.time() * 1000)).encode("utf-8")
        os.ftruncate(self.fd, 0)
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.write(self.fd, payload)
        os.fsync(self.fd)

    def _start_heartbeat(self) -> None:
        interval = min(max(self.config.heartbeat_s, 1), 2)

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self._write_heartbeat()
                except Exception:
                    return

        self._heartbeat_thread = threading.Thread(target=_loop, name="gpulock-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _register_signal_cleanup(self) -> None:
        def _handler(signum, _frame):
            self.release()
            os._exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            self._old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)

        if not self._registered_atexit:
            atexit.register(self.release)
            self._registered_atexit = True

    def _restore_signal_handlers(self) -> None:
        for sig, old in self._old_handlers.items():
            try:
                signal.signal(sig, old)
            except Exception:
                pass
        self._old_handlers.clear()

    def _reader_paths_locked(self) -> list[Path]:
        if not self.readers_dir.exists():
            return []
        return sorted(self.readers_dir.glob("*.lock"))

    def _next_queue_seq_locked(self) -> int:
        current = 0
        try:
            raw = self.queue_seq_path.read_text(encoding="utf-8").strip()
            if raw:
                current = int(raw)
        except Exception:
            current = 0
        seq = current + 1
        tmp = self.queue_seq_path.with_name(f".queue.seq.{os.getpid()}")
        tmp.write_text(f"{seq}\n", encoding="utf-8")
        os.replace(tmp, self.queue_seq_path)
        return seq

    def _register_queue_request_locked(self) -> None:
        if self._queue_request_path is not None and self._queue_request_path.exists():
            return

        seq = self._next_queue_seq_locked()
        rid = uuid.uuid4().hex[:10]
        req_path = self.queue_dir / f"req-{seq:020d}-{os.getpid()}-{rid}.req"
        cmdline = " ".join(shlex.quote(x) for x in sys.argv)
        payload = (
            f"pid={os.getpid()}\n"
            f"mode={self.mode}\n"
            f"seq={seq}\n"
            f"start_time={time.time():.6f}\n"
            f"cmdline={cmdline}\n"
        )
        fd = os.open(req_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._queue_request_path = req_path
        self._queue_request_seq = seq

    def _unregister_queue_request_locked(self) -> None:
        if self._queue_request_path is not None:
            self._queue_request_path.unlink(missing_ok=True)
        self._queue_request_path = None
        self._queue_request_seq = None

    def _queue_entries_locked(self) -> list[tuple[int, str, int, Path]]:
        entries: list[tuple[int, str, int, Path]] = []
        for req_path in self.queue_dir.glob("*.req"):
            meta = read_lock_metadata(req_path)
            mode = meta.get("mode", "").strip().lower()
            if mode not in (READ_MODE, WRITE_MODE):
                req_path.unlink(missing_ok=True)
                continue
            try:
                seq = int(meta.get("seq", "-1"))
                pid = int(meta.get("pid", "-1"))
            except ValueError:
                req_path.unlink(missing_ok=True)
                continue
            if seq <= 0 or pid <= 0:
                req_path.unlink(missing_ok=True)
                continue
            entries.append((seq, mode, pid, req_path))
        entries.sort(key=lambda x: (x[0], str(x[3])))
        return entries

    def _cleanup_stale_queue_locked(self) -> None:
        for _seq, _mode, pid, req_path in self._queue_entries_locked():
            if pid_exists(pid):
                continue
            req_path.unlink(missing_ok=True)

    def _try_cleanup_zombie_lock(self, lock_path: Path, now_s: float, log: logging.Logger) -> None:
        try:
            st = lock_path.stat()
        except FileNotFoundError:
            return

        age_s = now_s - st.st_mtime
        min_age_s = max(self.config.heartbeat_s * 2, 2)
        if age_s < min_age_s:
            return

        pid = read_lock_pid(lock_path)
        if pid is None:
            if age_s < self.config.grace_age_s:
                return
        else:
            if pid_exists(pid):
                return

        try:
            st2 = lock_path.stat()
        except FileNotFoundError:
            return
        if st2.st_mtime_ns != st.st_mtime_ns:
            return

        try:
            lock_path.unlink()
        except FileNotFoundError:
            return
        log.warning(
            "cleaned zombie lock gpu=%d path=%s pid=%s age_s=%.1f",
            self.physical_device_id,
            str(lock_path),
            str(pid) if pid is not None else "?",
            age_s,
        )

    def _cleanup_zombie_locks_locked(self) -> None:
        log = logging.getLogger("gpulock.main")
        now_s = time.time()
        candidates: list[Path] = []
        if self.writer_path.exists():
            candidates.append(self.writer_path)
        candidates.extend(self._reader_paths_locked())
        for lock_path in candidates:
            self._try_cleanup_zombie_lock(lock_path, now_s, log)

    def _queue_wait_reason_locked(self) -> Optional[str]:
        if self._queue_request_seq is None:
            return "queue request missing"
        if self._queue_request_path is None or not self._queue_request_path.exists():
            return "queue request disappeared"

        my_seq = self._queue_request_seq
        entries = self._queue_entries_locked()
        earlier = [item for item in entries if item[0] < my_seq]
        if self.mode == WRITE_MODE:
            if earlier:
                return f"queue waiting: {len(earlier)} earlier request(s)"
            return None

        earlier_writers = [item for item in earlier if item[1] == WRITE_MODE]
        if earlier_writers:
            return f"queue waiting: {len(earlier_writers)} earlier writer request(s)"
        return None

    def _reset_probe_state(self, lock_path: Path) -> None:
        self._orphan_probe.pop(str(lock_path), None)

    def _try_cleanup_orphan(self, lock_path: Path, now_s: float) -> None:
        try:
            st = lock_path.stat()
        except FileNotFoundError:
            self._reset_probe_state(lock_path)
            return

        age_s = now_s - st.st_mtime
        if age_s <= self.config.grace_age_s:
            self._reset_probe_state(lock_path)
            return

        pid = read_lock_pid(lock_path)
        if pid is not None and pid_exists(pid):
            self._reset_probe_state(lock_path)
            return

        key = str(lock_path)
        state = self._orphan_probe.setdefault(key, ProbeState())
        if now_s - state.last_probe_s < self.config.orphan_check_s:
            return
        state.last_probe_s = now_s

        if gpu_has_processes_by_index(self.physical_device_id):
            state.empty_count = 0
            state.last_mtime_ns = -1
            state.last_hb_ms = -1
            return

        hb = read_last_heartbeat_ms(lock_path)
        mtime_ns = st.st_mtime_ns

        if hb == state.last_hb_ms and mtime_ns == state.last_mtime_ns:
            state.empty_count += 1
        else:
            state.empty_count = 1
            state.last_hb_ms = hb
            state.last_mtime_ns = mtime_ns

        if state.empty_count < self.config.orphan_empty_threshold:
            return

        try:
            st2 = lock_path.stat()
        except FileNotFoundError:
            self._reset_probe_state(lock_path)
            return

        hb2 = read_last_heartbeat_ms(lock_path)
        age2 = time.time() - st2.st_mtime
        pid2 = read_lock_pid(lock_path)

        if (
            age2 > self.config.grace_age_s
            and hb2 == state.last_hb_ms
            and st2.st_mtime_ns == state.last_mtime_ns
            and (pid2 is None or not pid_exists(pid2))
            and not gpu_has_processes_by_index(self.physical_device_id)
        ):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

        self._reset_probe_state(lock_path)

    def _cleanup_orphans_locked(self) -> None:
        now_s = time.time()
        candidates: list[Path] = []
        if self.writer_path.exists():
            candidates.append(self.writer_path)
        candidates.extend(self._reader_paths_locked())

        seen = {str(p) for p in candidates}
        for stale_key in list(self._orphan_probe.keys()):
            if stale_key not in seen:
                self._orphan_probe.pop(stale_key, None)

        for lock_path in candidates:
            self._try_cleanup_orphan(lock_path, now_s)

    def _acquire_write_locked(self) -> Optional[str]:
        if self.writer_path.exists():
            return f"writer lock exists: {self.writer_path}"
        readers = self._reader_paths_locked()
        if readers:
            return f"{len(readers)} reader lock(s) active"

        self.lock_path = self.writer_path
        self.fd = os.open(self.writer_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._write_heartbeat()
        return None

    def _acquire_read_locked(self) -> Optional[str]:
        if self.writer_path.exists():
            return f"writer lock exists: {self.writer_path}"

        rid = uuid.uuid4().hex[:12]
        self.lock_path = self.readers_dir / f"reader-{os.getpid()}-{rid}.lock"
        self.fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._write_heartbeat()
        return None

    def _ensure_write_lock_gpu_ready(self) -> None:
        if self.mode != WRITE_MODE:
            return

        zero_util_streak = 0
        busy_streak = 0
        deadline = time.monotonic() + self.config.timeout_s
        last_reason = ""
        while True:
            busy, reason = gpu_busy_reason_for_perf(self.physical_device_id)
            now = time.monotonic()
            last_reason = reason

            if busy:
                zero_util_streak = 0
                busy_streak += 1
                if not self.wait_gpu_idle:
                    if busy_streak < 2:
                        time.sleep(self.idle_check_ms / 1000.0)
                        continue
                    raise RuntimeError(
                        f"GPU{self.physical_device_id} appears busy before write lock ({reason}). "
                        "Running perf with write lock may be inaccurate while other workloads are active. "
                        "You can run correctness first with a read lock (gpulock check ...), "
                        f"or add --wait-gpu-idle to wait for {self.idle_streak_s} consecutive util=0 checks."
                    )
            else:
                busy_streak = 0
                zero_util_streak += 1
                if zero_util_streak >= self.idle_streak_s:
                    return

            if now > deadline:
                raise TimeoutError(
                    f"Timeout while waiting for GPU{self.physical_device_id} to reach "
                    f"{self.idle_streak_s} consecutive util=0 checks before write lock "
                    f"(last_state: {last_reason}, >{self.config.timeout_s}s)"
                )
            time.sleep(self.idle_check_ms / 1000.0)

    def acquire(self) -> None:
        log = logging.getLogger("gpulock.main")
        notify_guard_activity(self.root, self.physical_device_id, self.mode)
        if park_placeholder(self.gpu_dir, timeout_s=5.0):
            log.info("gpu%d: parked placeholder worker before lock acquire", self.physical_device_id)
        else:
            kill_placeholder(self.gpu_dir)
            kill_visible_placeholder_compute_pids(self.physical_device_id)
        self._ensure_write_lock_gpu_ready()
        acquired = False
        deadline = time.monotonic() + self.config.timeout_s
        last_blocked_reason = ""
        try:
            while True:
                blocked_reason = "unknown"
                with self._state_gate():
                    self._register_queue_request_locked()
                    self._cleanup_stale_queue_locked()
                    self._cleanup_zombie_locks_locked()
                    self._cleanup_orphans_locked()

                    queue_reason = self._queue_wait_reason_locked()
                    if queue_reason is not None:
                        blocked_reason = queue_reason
                    else:
                        try:
                            if self.mode == WRITE_MODE:
                                blocked_reason = self._acquire_write_locked() or ""
                            else:
                                blocked_reason = self._acquire_read_locked() or ""
                        except FileExistsError:
                            blocked_reason = "lock file race, retry"

                    if blocked_reason == "":
                        self._unregister_queue_request_locked()
                        self._start_heartbeat()
                        self._register_signal_cleanup()
                        log.info(
                            "lock acquired gpu=%d mode=%s lock_path=%s",
                            self.physical_device_id,
                            self.mode,
                            str(self.lock_path) if self.lock_path is not None else "",
                        )
                        acquired = True
                        return

                if blocked_reason != "" and blocked_reason != last_blocked_reason:
                    last_blocked_reason = blocked_reason
                    log.info(
                        "lock waiting gpu=%d mode=%s reason=%s",
                        self.physical_device_id,
                        self.mode,
                        blocked_reason,
                    )

                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timeout while waiting for {self.mode} lock on GPU{self.physical_device_id}: {blocked_reason} "
                        f"(>{self.config.timeout_s}s)"
                    )
                time.sleep(max(self.config.poll_ms, 1) / 1000.0)
        finally:
            if not acquired:
                with contextlib.suppress(Exception):
                    with self._state_gate():
                        self._unregister_queue_request_locked()

    def release(self) -> None:
        if self.fd is None or self.lock_path is None:
            return
        log = logging.getLogger("gpulock.main")
        released_path = str(self.lock_path)
        released_mode = self.mode
        released_gpu = self.physical_device_id

        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=3.0)

        try:
            os.close(self.fd)
        except Exception:
            pass

        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

        self.fd = None
        self.lock_path = None
        self._restore_signal_handlers()
        log.info("lock released gpu=%d mode=%s lock_path=%s", released_gpu, released_mode, released_path)

    def __enter__(self) -> "GpuBenchLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

"""Built-in supervisor backend for environments without systemd.

This module implements two flavours of operation:

* The user-facing actions (``install``, ``start``, ``stop``, ``status``,
  ``logs``, ...). These are short-lived commands that manage a separate,
  daemonised supervisor process.
* The supervisor process itself, entered via
  ``gpulock service _run-supervisor``. It double-forks to detach from the
  controlling terminal, then loops forever spawning ``gpulock guard`` as a
  child and restarting it (with backoff) on crash.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..gpu import pid_exists
from ..paths import resolve_lock_root
from .common import GuardServiceConfig, service_dir


SUPERVISOR_PID_FILENAME = "supervisor.pid"
GUARD_PID_FILENAME = "service-guard.pid"
SUPERVISOR_LOG_FILENAME = "supervisor.log"


def supervisor_pid_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / SUPERVISOR_PID_FILENAME


def guard_pid_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / GUARD_PID_FILENAME


def supervisor_log_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / SUPERVISOR_LOG_FILENAME


def _read_pid(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def supervisor_running(lock_root: Path | None = None) -> tuple[bool, int]:
    pid = _read_pid(supervisor_pid_path(lock_root))
    if pid <= 0:
        return (False, 0)
    return (pid_exists(pid), pid)


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

def install(cfg: GuardServiceConfig, *, start_now: bool = True) -> Path:
    cfg.save()  # config persisted by caller usually, but keep it idempotent
    if start_now:
        start()
    return service_dir() / "config.json"


def uninstall() -> None:
    stop()
    cfg_path = service_dir() / "config.json"
    cfg_path.unlink(missing_ok=True)


def start() -> int:
    """Spawn the supervisor as a detached background process."""
    running, pid = supervisor_running()
    if running:
        print(f"[gpulock service] supervisor already running (pid={pid})")
        return 0

    # Stale PID file -> remove.
    supervisor_pid_path().unlink(missing_ok=True)

    python = sys.executable or "python3"
    cmd = [python, "-m", "gpulock", "service", "_run-supervisor"]
    log_path = supervisor_log_path()
    log_path.touch(exist_ok=True)
    try:
        os.chmod(log_path, 0o600)
    except PermissionError:
        pass

    log_fp = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=log_fp,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        log_fp.close()

    # Wait briefly for the supervisor to write its PID file.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        running, pid = supervisor_running()
        if running:
            print(f"[gpulock service] supervisor started (pid={pid}, log={log_path})")
            return 0
        if proc.poll() is not None:
            print(
                "[gpulock service] supervisor exited immediately. "
                f"check {log_path} for details.",
                file=sys.stderr,
            )
            return proc.returncode or 1
        time.sleep(0.1)
    print(
        "[gpulock service] supervisor did not write a PID file in time; "
        f"check {log_path}.",
        file=sys.stderr,
    )
    return 1


def stop(timeout_s: float = 15.0) -> int:
    running, pid = supervisor_running()
    if not running:
        supervisor_pid_path().unlink(missing_ok=True)
        guard_pid_path().unlink(missing_ok=True)
        print("[gpulock service] supervisor not running")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        if e.errno == errno.ESRCH:
            supervisor_pid_path().unlink(missing_ok=True)
            return 0
        print(f"[gpulock service] failed to signal supervisor pid={pid}: {e}", file=sys.stderr)
        return 1

    deadline = time.monotonic() + max(timeout_s, 1.0)
    while time.monotonic() < deadline:
        if not pid_exists(pid):
            supervisor_pid_path().unlink(missing_ok=True)
            guard_pid_path().unlink(missing_ok=True)
            print(f"[gpulock service] supervisor stopped (pid={pid})")
            return 0
        time.sleep(0.2)

    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
    supervisor_pid_path().unlink(missing_ok=True)
    guard_pid_path().unlink(missing_ok=True)
    print(f"[gpulock service] supervisor force-killed (pid={pid})", file=sys.stderr)
    return 0


def restart() -> int:
    stop()
    return start()


def status() -> int:
    running, sup_pid = supervisor_running()
    guard_pid = _read_pid(guard_pid_path())
    guard_alive = guard_pid > 0 and pid_exists(guard_pid)
    print(f"backend: supervisor")
    print(f"supervisor: {'running' if running else 'stopped'} (pid={sup_pid or '-'})")
    print(f"guard:      {'running' if guard_alive else 'stopped'} (pid={guard_pid or '-'})")
    log_path = supervisor_log_path()
    if log_path.exists():
        try:
            size = log_path.stat().st_size
        except OSError:
            size = 0
        print(f"log:        {log_path} ({size} bytes)")
    return 0 if running and guard_alive else 3


def logs(*, lines: int = 200, follow: bool = False) -> int:
    log_path = supervisor_log_path()
    if not log_path.exists():
        print(f"[gpulock service] no supervisor log at {log_path}", file=sys.stderr)
        return 1
    args = ["tail", "-n", str(max(lines, 1))]
    if follow:
        args.append("-F")
    args.append(str(log_path))
    try:
        return subprocess.call(args)
    except FileNotFoundError:
        # Fallback: tiny in-Python tail
        try:
            data = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[gpulock service] failed to read {log_path}: {e}", file=sys.stderr)
            return 1
        for line in data.splitlines()[-max(lines, 1):]:
            print(line)
        return 0


# ---------------------------------------------------------------------------
# Supervisor entry point (long-running)
# ---------------------------------------------------------------------------

def _setup_supervisor_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("gpulock.service.supervisor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [supervisor] pid=%(process)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(str(log_path), maxBytes=20 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _daemonize() -> None:
    """Standard double-fork to detach from the controlling terminal."""
    if os.getenv("GPULOCK_SUPERVISOR_FOREGROUND", "").strip().lower() in ("1", "true", "yes", "on"):
        return

    pid = os.fork()
    if pid > 0:
        os._exit(0)
    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.umask(0o077)
    try:
        os.chdir("/")
    except OSError:
        pass

    devnull_r = os.open(os.devnull, os.O_RDONLY)
    devnull_w = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_r, 0)
        os.dup2(devnull_w, 1)
        os.dup2(devnull_w, 2)
    finally:
        os.close(devnull_r)
        os.close(devnull_w)


def run_supervisor() -> int:
    """Long-running entry point: supervises ``gpulock guard``."""
    lock_root = resolve_lock_root()
    log = _setup_supervisor_logger(supervisor_log_path(lock_root))

    try:
        cfg = GuardServiceConfig.load(lock_root)
    except FileNotFoundError as e:
        print(f"[gpulock service] {e}", file=sys.stderr)
        log.error("missing service config: %s", e)
        return 1

    _daemonize()

    pid_path = supervisor_pid_path(lock_root)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        os.chmod(pid_path, 0o600)
    except PermissionError:
        pass
    log.info("supervisor started; lock_root=%s gpu_ids=%s", lock_root, cfg.gpu_ids)

    stopping = False
    child: subprocess.Popen | None = None

    def kill_child(sig: int = signal.SIGTERM, *, wait_s: float = 10.0) -> None:
        nonlocal child
        if child is None:
            return
        if child.poll() is None:
            with contextlib.suppress(OSError):
                child.send_signal(sig)
            try:
                child.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    child.kill()
                with contextlib.suppress(Exception):
                    child.wait(timeout=5)
        guard_pid_path(lock_root).unlink(missing_ok=True)
        child = None

    def on_term(_sig, _frame):
        nonlocal stopping
        stopping = True
        kill_child(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    backoff_s = 1.0
    backoff_max_s = 60.0
    try:
        while not stopping:
            argv: list[str] = []
            binary = cfg.gpulock_executable.strip()
            if binary:
                argv = [binary, *cfg.to_guard_argv()]
            else:
                python = cfg.python_executable.strip() or sys.executable
                argv = [python, "-m", "gpulock", *cfg.to_guard_argv()]

            log.info("spawning guard: %s", " ".join(argv))
            env = os.environ.copy()
            env.update(cfg.extra_env)

            try:
                child = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
            except FileNotFoundError as e:
                log.error("guard executable missing: %s", e)
                time.sleep(min(backoff_s, backoff_max_s))
                backoff_s = min(backoff_s * 2, backoff_max_s)
                continue

            guard_pid_path(lock_root).write_text(str(child.pid), encoding="utf-8")
            try:
                os.chmod(guard_pid_path(lock_root), 0o600)
            except PermissionError:
                pass
            start_ts = time.monotonic()

            rc = None
            while not stopping:
                try:
                    rc = child.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if stopping:
                kill_child(signal.SIGTERM)
                break

            guard_pid_path(lock_root).unlink(missing_ok=True)

            ran_for = time.monotonic() - start_ts
            log.warning("guard exited rc=%s after %.1fs", rc, ran_for)
            if ran_for >= 30.0:
                backoff_s = 1.0
            sleep_s = min(backoff_s, backoff_max_s)
            log.info("restarting guard in %.1fs", sleep_s)
            for _ in range(int(sleep_s * 10)):
                if stopping:
                    break
                time.sleep(0.1)
            backoff_s = min(backoff_s * 2, backoff_max_s)
    finally:
        kill_child(signal.SIGTERM)
        with contextlib.suppress(Exception):
            if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink(missing_ok=True)
        log.info("supervisor exiting")
    return 0

"""supervisord backend for the gpulock guard service.

Wraps the third-party ``supervisor`` package. All state lives under
``${lock_root}/service/``:

* ``config.json``       gpulock-managed runtime config (see common.py)
* ``supervisord.conf``  generated; rewritten on every ``service start/restart``
* ``supervisord.pid``   supervisord's own pid file
* ``supervisord.log``   supervisord's own log
* ``supervisor.sock``   unix domain socket for supervisorctl
* ``guard.log``         combined stdout+stderr of ``gpulock guard``
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..gpu import pid_exists
from ..placeholder import placeholder_socket_path, stop_placeholder
from .common import GuardServiceConfig, chmod_quiet, guard_status_path, say, service_dir, warn


PROGRAM_NAME = "gpulock-guard"
CONF_FILENAME = "supervisord.conf"
PID_FILENAME = "supervisord.pid"
SOCK_FILENAME = "supervisor.sock"
SUPERVISORD_LOG_FILENAME = "supervisord.log"
GUARD_LOG_FILENAME = "guard.log"

# Always invoke supervisord/supervisorctl through the same interpreter that
# imports gpulock, so we hit the gpulock-managed venv even when the user has
# multiple Pythons in PATH.
_SUPERVISORD = [sys.executable, "-m", "supervisor.supervisord"]
_SUPERVISORCTL = [sys.executable, "-m", "supervisor.supervisorctl"]


# --- paths ---------------------------------------------------------------

def conf_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / CONF_FILENAME


def pid_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / PID_FILENAME


def sock_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / SOCK_FILENAME


def supervisord_log_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / SUPERVISORD_LOG_FILENAME


def guard_log_path(lock_root: Path | None = None) -> Path:
    return service_dir(lock_root) / GUARD_LOG_FILENAME


# --- low-level state queries ---------------------------------------------

def _read_pid(path: Path) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return 0
    try:
        return max(int(raw or 0), 0)
    except ValueError:
        return 0


def running_pid(lock_root: Path | None = None) -> int:
    """Return supervisord's pid if alive, else 0."""
    pid = _read_pid(pid_path(lock_root))
    return pid if pid > 0 and pid_exists(pid) else 0


def supervisor_available() -> tuple[bool, str]:
    """Check that the third-party `supervisor` package is importable."""
    try:
        import supervisor  # noqa: F401
    except ImportError as e:
        return (False, str(e))
    return (True, "")


def _cleanup_stale() -> None:
    pid_path().unlink(missing_ok=True)
    sock_path().unlink(missing_ok=True)


def _cleanup_placeholder_workers(timeout_s: float = 5.0) -> None:
    """Best-effort cleanup for placeholder workers left under this lock root.

    The guard starts placeholders in their own sessions so a hard supervisor
    shutdown does not necessarily kill them as part of the guard process group.
    `service stop/restart/uninstall` should still leave no service-owned
    placeholder behind, including GPUs removed from the new config.
    """
    root = service_dir().parent
    for pid_file in sorted(root.glob("gpu*/placeholder.pid")):
        gpu_dir = pid_file.parent
        with contextlib.suppress(Exception):
            stop_placeholder(gpu_dir, timeout_s=1.0)

        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
            placeholder_socket_path(gpu_dir).unlink(missing_ok=True)
            continue

        deadline = time.monotonic() + max(timeout_s, 0.1)
        while time.monotonic() < deadline and pid_exists(pid):
            time.sleep(0.1)
        if pid_exists(pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
        if pid_exists(pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

        pid_file.unlink(missing_ok=True)
        placeholder_socket_path(gpu_dir).unlink(missing_ok=True)


# --- conf rendering ------------------------------------------------------

def _format_environment(env: dict[str, str]) -> str:
    """Format an env dict for supervisord's ``environment=KEY="VAL",...`` syntax.

    Raises ``ValueError`` if a value contains a character that supervisord's
    quoting cannot represent (double quote / newline).
    """
    parts: list[str] = []
    for k, v in env.items():
        if any(c in v for c in ('"', "\n", "\r")):
            raise ValueError(
                f"extra_env value for {k!r} contains an unsupported character "
                "(double-quote or newline). edit supervisord.conf manually if you "
                "really need this."
            )
        parts.append(f'{k}="{v}"')
    return ",".join(parts)


def _build_program_command(cfg: GuardServiceConfig) -> str:
    binary = cfg.gpulock_executable.strip()
    if binary:
        argv = [binary, *cfg.to_guard_argv()]
    else:
        python = cfg.python_executable.strip() or sys.executable
        argv = [python, "-m", "gpulock", *cfg.to_guard_argv()]
    return " ".join(shlex.quote(a) for a in argv)


def render_conf(cfg: GuardServiceConfig) -> str:
    """Render the supervisord.conf body.

    Uses ``%(here)s`` everywhere so the file is location-independent: the
    service directory can be moved or symlinked without touching the conf.
    """
    cmd = _build_program_command(cfg)
    env_line = _format_environment(cfg.extra_env)
    env_section = f"environment={env_line}\n" if env_line else ""
    return f"""\
; gpulock-managed supervisord config. regenerated on every `service start/restart`.
; do NOT edit by hand; use `gpulock service config ...` instead.

[unix_http_server]
file=%(here)s/{SOCK_FILENAME}
chmod=0700

[supervisord]
logfile=%(here)s/{SUPERVISORD_LOG_FILENAME}
logfile_maxbytes=20MB
logfile_backups=5
pidfile=%(here)s/{PID_FILENAME}
nodaemon=false
silent=false

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://%(here)s/{SOCK_FILENAME}

[program:{PROGRAM_NAME}]
command={cmd}
autostart=true
autorestart=true
startsecs=3
startretries=10
stopwaitsecs=10
stopsignal=TERM
killasgroup=true
stopasgroup=true
redirect_stderr=true
stdout_logfile=%(here)s/{GUARD_LOG_FILENAME}
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
{env_section}"""


def write_conf(cfg: GuardServiceConfig, lock_root: Path | None = None) -> Path:
    path = conf_path(lock_root)
    path.write_text(render_conf(cfg), encoding="utf-8")
    chmod_quiet(path, 0o600)
    return path


# --- supervisorctl plumbing ----------------------------------------------

def _supervisorctl(*args: str, timeout_s: float = 30.0) -> tuple[int, str, str]:
    cmd = [*_SUPERVISORCTL, "-c", str(conf_path()), *args]
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout_s,
        )
    except FileNotFoundError as e:
        return (127, "", f"supervisorctl not available: {e}\n")
    return (proc.returncode, proc.stdout, proc.stderr)


# --- user-facing actions -------------------------------------------------

def install(cfg: GuardServiceConfig, *, start_now: bool) -> int:
    cfg.save()
    write_conf(cfg)
    return start() if start_now else 0


def start() -> int:
    """Launch supervisord in the background. Idempotent."""
    ok, err = supervisor_available()
    if not ok:
        warn(
            f"supervisor package not importable: {err}\n"
            "  reinstall gpulock to pull in the dependency:\n"
            "    uv tool install -e . --force --reinstall --refresh --torch-backend auto"
        )
        return 1

    if pid := running_pid():
        say(f"supervisord already running (pid={pid})")
        return 0
    _cleanup_stale()

    try:
        cfg = GuardServiceConfig.load()
    except FileNotFoundError as e:
        warn(str(e))
        return 1
    try:
        write_conf(cfg)
    except ValueError as e:
        warn(str(e))
        return 2

    cmd = [*_SUPERVISORD, "-c", str(conf_path())]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        warn("supervisord did not daemonize within 15s")
        return 1
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pid := running_pid():
            say(f"supervisord started (pid={pid}, conf={conf_path()})")
            return 0
        time.sleep(0.1)

    warn(f"supervisord did not write a pid file in time; check {supervisord_log_path()}.")
    return 1


def stop(timeout_s: float = 30.0) -> int:
    pid = running_pid()
    if pid == 0:
        _cleanup_placeholder_workers()
        _cleanup_stale()
        say("supervisord not running")
        return 0

    rc, out, err = _supervisorctl("shutdown", timeout_s=timeout_s)
    if out.strip():
        print(out.strip())
    if rc != 0 and err.strip():
        print(err.strip(), file=sys.stderr)

    deadline = time.monotonic() + max(timeout_s, 1.0)
    while time.monotonic() < deadline:
        if not pid_exists(pid):
            _cleanup_placeholder_workers()
            _cleanup_stale()
            say(f"supervisord stopped (pid={pid})")
            return 0
        time.sleep(0.1)

    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
    _cleanup_placeholder_workers()
    _cleanup_stale()
    warn(f"supervisord force-killed (pid={pid})")
    return 0


def restart() -> int:
    rc = stop()
    return rc if rc != 0 else start()


def _format_optional_seconds(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "unknown"


def _print_guard_snapshot() -> None:
    path = guard_status_path()
    if not path.exists():
        print(f"guard status: missing ({path})")
        print("guard detail: no guard snapshot yet; check guard log if the service just started")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"guard status: unreadable ({path}: {e})")
        return

    updated = data.get("updated_at")
    age_text = "unknown age"
    stale_text = ""
    try:
        age_s = max(time.time() - float(updated), 0.0)
        age_text = f"{age_s:.1f}s ago"
        if age_s > 10.0:
            stale_text = " (stale)"
    except (TypeError, ValueError):
        pass
    updated_text = str(data.get("updated_at_text") or "<unknown>")
    print(f"guard status: updated {age_text}{stale_text} ({updated_text})")

    gpus = data.get("gpus", [])
    if not isinstance(gpus, list) or not gpus:
        print("gpu status:   <none>")
        return

    print("gpu status:")
    for item in gpus:
        if not isinstance(item, dict):
            continue
        gpu_id = item.get("gpu_id", "?")
        placeholder = item.get("placeholder", "unknown")
        pid = item.get("placeholder_pid")
        pid_text = f" pid={pid}" if pid else ""
        reason = item.get("reason", "unknown")
        runtime = item.get("runtime")
        runtime_text = ""
        if isinstance(runtime, dict):
            runtime_text = (
                f" util={runtime.get('util_gpu', '?')}%"
                f" mem={runtime.get('mem_used_mib', '?')}/{runtime.get('mem_total_mib', '?')}MiB"
                f" compute_pids={runtime.get('visible_compute_pids', '?')}"
                f" non_placeholder_pids={runtime.get('visible_non_placeholder_pids', '?')}"
            )
            if "known_placeholder_compute_pids" in runtime:
                runtime_text += f" known_placeholder_pids={runtime.get('known_placeholder_compute_pids', [])}"
            if "external_compute_pids" in runtime:
                runtime_text += f" external_pids={runtime.get('external_compute_pids', [])}"
        idle_for = item.get("idle_for_s")
        last_gpulock_activity_age = item.get("last_gpulock_activity_age_s", item.get("last_activity_age_s"))
        last_user_gpu_activity_age = item.get("last_user_gpu_activity_age_s")
        timing_text = (
            f" idle_for={_format_optional_seconds(idle_for)}"
            f" last_gpulock_activity={_format_optional_seconds(last_gpulock_activity_age)}"
            f" last_user_gpu_activity={_format_optional_seconds(last_user_gpu_activity_age)}"
        )
        print(
            f"  gpu{gpu_id}: placeholder={placeholder}{pid_text};"
            f"{runtime_text}{timing_text}; reason={reason}"
        )


def status() -> int:
    """Print high-level status. Exit codes follow the LSB / systemd convention:

    * 0 — installed and supervisord+guard running healthily
    * 3 — installed, supervisord stopped (or guard not RUNNING)
    * 4 — not installed (no config.json)
    """
    cfg_path = GuardServiceConfig.config_path()
    if not cfg_path.exists():
        print(f"installed:    no  (missing config: {cfg_path})")
        print("next:         gpulock service install --no-start")
        return 4

    cfg = GuardServiceConfig.load()
    pid = running_pid()
    print(f"installed:    yes")
    print(f"config:       {cfg_path}")
    print(f"conf:         {conf_path()}")
    print(f"guard log:    {guard_log_path()}")
    print(f"gpu_ids:      {cfg.gpu_ids or '<all visible GPUs>'}")
    print(f"idle_timeout: {cfg.idle_timeout}s")
    print(f"guard_poll_s: {cfg.guard_poll_s}s")
    print(f"placeholder_release_mode: {cfg.placeholder_release_mode}")
    print(f"supervisord:  {'running (pid=' + str(pid) + ')' if pid else 'stopped'}")
    if pid == 0:
        return 3

    rc, out, err = _supervisorctl("status", PROGRAM_NAME)
    if out.strip():
        for line in out.strip().splitlines():
            print(f"program:      {line}")
    if rc != 0 and err.strip():
        print(err.strip(), file=sys.stderr)
    _print_guard_snapshot()
    return rc


def logs(*, lines: int = 200, follow: bool = False) -> int:
    log = guard_log_path()
    if not log.exists():
        warn(f"no guard log at {log}")
        return 1
    n = max(int(lines), 1)
    cmd = ["tail", "-n", str(n), *(["-F"] if follow else []), str(log)]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        try:
            data = log.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            warn(f"failed to read {log}: {e}")
            return 1
        for line in data.splitlines()[-n:]:
            print(line)
        return 0


def uninstall() -> int:
    rc = stop()
    GuardServiceConfig.config_path().unlink(missing_ok=True)
    conf_path().unlink(missing_ok=True)
    say("uninstalled")
    return rc

"""systemd --user backend for gpulock guard."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .common import SERVICE_NAME, GuardServiceConfig


UNIT_FILENAME = f"{SERVICE_NAME}.service"


def _user_unit_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME", "").strip()
    if base:
        unit_dir = Path(base) / "systemd" / "user"
    else:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    return unit_dir


def unit_path() -> Path:
    return _user_unit_dir() / UNIT_FILENAME


def _systemctl(args: Iterable[str], check: bool = False) -> subprocess.CompletedProcess:
    if shutil.which("systemctl") is None:
        raise RuntimeError(
            "systemctl not found in PATH; cannot manage systemd --user services. "
            "Try `gpulock service install --backend supervisor` instead."
        )
    cmd = ["systemctl", "--user", *args]
    return subprocess.run(cmd, check=check, text=True)


def _journalctl(args: Iterable[str]) -> int:
    if shutil.which("journalctl") is None:
        print(
            "[gpulock service] journalctl not found; cannot show systemd --user logs.",
            file=sys.stderr,
        )
        return 1
    cmd = ["journalctl", "--user", "-u", UNIT_FILENAME, *args]
    return subprocess.call(cmd)


def _exec_start_line(cfg: GuardServiceConfig) -> str:
    binary = cfg.gpulock_executable.strip()
    if binary:
        argv = [binary, *cfg.to_guard_argv()]
    else:
        python = cfg.python_executable.strip() or sys.executable
        argv = [python, "-m", "gpulock", *cfg.to_guard_argv()]
    return " ".join(shlex.quote(part) for part in argv)


def render_unit(cfg: GuardServiceConfig) -> str:
    env_lines: list[str] = []
    for key, value in sorted(cfg.extra_env.items()):
        env_lines.append(f'Environment="{key}={value}"')
    env_block = "\n".join(env_lines)
    if env_block:
        env_block += "\n"

    return (
        "[Unit]\n"
        "Description=gpulock guard daemon (per-user GPU placeholder watchdog)\n"
        "After=default.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"{env_block}"
        f"ExecStart={_exec_start_line(cfg)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "TimeoutStopSec=20\n"
        "KillMode=mixed\n"
        "KillSignal=SIGTERM\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install(cfg: GuardServiceConfig, *, start: bool = True, enable: bool = True) -> Path:
    path = unit_path()
    path.write_text(render_unit(cfg), encoding="utf-8")
    try:
        os.chmod(path, 0o644)
    except PermissionError:
        pass
    _systemctl(["daemon-reload"])
    if enable:
        _systemctl(["enable", UNIT_FILENAME], check=False)
    if start:
        _systemctl(["restart", UNIT_FILENAME], check=False)
    return path


def uninstall() -> None:
    path = unit_path()
    _systemctl(["disable", "--now", UNIT_FILENAME], check=False)
    if path.exists():
        path.unlink(missing_ok=True)
    _systemctl(["daemon-reload"], check=False)


def start() -> int:
    return _systemctl(["restart", UNIT_FILENAME]).returncode


def stop() -> int:
    return _systemctl(["stop", UNIT_FILENAME]).returncode


def restart() -> int:
    return _systemctl(["restart", UNIT_FILENAME]).returncode


def status() -> int:
    return _systemctl(["status", UNIT_FILENAME, "--no-pager"]).returncode


def enable() -> int:
    return _systemctl(["enable", UNIT_FILENAME]).returncode


def disable() -> int:
    return _systemctl(["disable", UNIT_FILENAME]).returncode


def logs(*, lines: int = 200, follow: bool = False) -> int:
    args: list[str] = ["-n", str(max(lines, 1))]
    if follow:
        args.append("-f")
    return _journalctl(args)

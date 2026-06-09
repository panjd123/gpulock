"""Shared helpers for the gpulock guard service."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import resolve_lock_root


_PREFIX = "[gpulock service]"
DEFAULT_IDLE_TIMEOUT = 5400
DEFAULT_PLACEHOLDER_IDLE_S = 1.0
GUARD_STATUS_FILENAME = "guard.status.json"


def say(msg: str) -> None:
    """stdout message with the [gpulock service] prefix."""
    print(f"{_PREFIX} {msg}")


def warn(msg: str) -> None:
    """stderr message with the [gpulock service] prefix."""
    print(f"{_PREFIX} {msg}", file=sys.stderr)


def chmod_quiet(path: Path, mode: int) -> None:
    """Best-effort chmod that ignores PermissionError (e.g. on shared mounts)."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def service_dir(lock_root: Path | None = None) -> Path:
    """Directory holding service config / pid / log files."""
    root = lock_root or resolve_lock_root()
    path = root / "service"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def guard_status_path(lock_root: Path | None = None) -> Path:
    """Path to the guard's machine-readable status snapshot."""
    return service_dir(lock_root) / GUARD_STATUS_FILENAME


@dataclass
class GuardServiceConfig:
    """Persistent runtime configuration for the guard service.

    Stored at ``${lock_root}/service/config.json``. Owned by gpulock; the
    generated ``supervisord.conf`` is regenerated from this on every
    ``service start/restart``.
    """

    gpu_ids: list[int] = field(default_factory=list)
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT
    placeholder_idle_s: float = DEFAULT_PLACEHOLDER_IDLE_S
    extra_env: dict[str, str] = field(default_factory=dict)
    python_executable: str = ""
    gpulock_executable: str = ""

    def to_guard_argv(self) -> list[str]:
        return [
            "guard",
            *(str(g) for g in self.gpu_ids),
            "--idle-timeout", str(self.idle_timeout),
            "--placeholder-idle-s", str(self.placeholder_idle_s),
        ]

    def save(self, lock_root: Path | None = None) -> Path:
        path = self.config_path(lock_root)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        chmod_quiet(path, 0o600)
        return path

    @classmethod
    def load(cls, lock_root: Path | None = None) -> "GuardServiceConfig":
        path = cls.config_path(lock_root)
        if not path.exists():
            raise FileNotFoundError(
                f"no gpulock service config found at {path}. "
                "run `gpulock service install` first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            gpu_ids=[int(x) for x in data.get("gpu_ids", [])],
            idle_timeout=int(data.get("idle_timeout", DEFAULT_IDLE_TIMEOUT)),
            placeholder_idle_s=float(data.get("placeholder_idle_s", DEFAULT_PLACEHOLDER_IDLE_S)),
            extra_env={str(k): str(v) for k, v in dict(data.get("extra_env", {})).items()},
            python_executable=str(data.get("python_executable", "")),
            gpulock_executable=str(data.get("gpulock_executable", "")),
        )

    @classmethod
    def config_path(cls, lock_root: Path | None = None) -> Path:
        return service_dir(lock_root) / "config.json"

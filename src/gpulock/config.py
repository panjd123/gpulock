"""Shared constants, env helpers and dataclasses for gpulock."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


READ_MODE = "read"
WRITE_MODE = "write"
PLACEHOLDER_RELEASE_PARK = "park"
PLACEHOLDER_RELEASE_STOP = "stop"
PLACEHOLDER_RELEASE_MODES = (PLACEHOLDER_RELEASE_STOP, PLACEHOLDER_RELEASE_PARK)
DEFAULT_PLACEHOLDER_RELEASE_MODE = PLACEHOLDER_RELEASE_STOP


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def normalize_placeholder_release_mode(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in ("", "clean", "full-stop", "fully-stop", "kill"):
        return PLACEHOLDER_RELEASE_STOP
    if value in ("legacy", "resident"):
        return PLACEHOLDER_RELEASE_PARK
    if value in PLACEHOLDER_RELEASE_MODES:
        return value
    raise ValueError(
        f"placeholder_release_mode must be one of "
        f"{', '.join(PLACEHOLDER_RELEASE_MODES)}, got {raw!r}"
    )


def _service_placeholder_release_mode_default() -> str:
    lock_root = os.getenv("GPULOCK_LOCK_DIR", "").strip()
    config_paths: list[Path] = []
    if lock_root:
        config_paths.append(Path(lock_root) / "service" / "config.json")
    else:
        config_paths.append(Path("/var/lock/gpulock/service/config.json"))
        config_paths.append(Path("/tmp/gpulock_locks/service/config.json"))

    for path in config_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            return normalize_placeholder_release_mode(
                data.get("placeholder_release_mode", DEFAULT_PLACEHOLDER_RELEASE_MODE)
            )
        except ValueError:
            continue
    return DEFAULT_PLACEHOLDER_RELEASE_MODE


@dataclass
class LockConfig:
    poll_ms: int = 200
    timeout_s: int = 1800
    grace_age_s: int = 180
    heartbeat_s: int = 2
    placeholder_release_mode: str = DEFAULT_PLACEHOLDER_RELEASE_MODE

    @classmethod
    def from_env(cls) -> "LockConfig":
        defaults = cls()
        placeholder_mode_default = _service_placeholder_release_mode_default()
        return cls(
            poll_ms=env_int("GPULOCK_POLL_MS", defaults.poll_ms),
            timeout_s=env_int("GPULOCK_TIMEOUT_S", defaults.timeout_s),
            grace_age_s=env_int("GPULOCK_GRACE_AGE_S", defaults.grace_age_s),
            heartbeat_s=env_int("GPULOCK_HEARTBEAT_S", defaults.heartbeat_s),
            placeholder_release_mode=normalize_placeholder_release_mode(
                os.getenv("GPULOCK_PLACEHOLDER_RELEASE_MODE", placeholder_mode_default)
            ),
        )


@dataclass
class StaleLockProbe:
    last_mtime_ns: int = -1
    last_hb_ms: int = -1


@dataclass
class GpuRuntimeState:
    util_gpu: int
    mem_used_mib: int
    mem_total_mib: int
    visible_compute_pids: int
    visible_non_placeholder_pids: int

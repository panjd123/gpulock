"""Shared constants, env helpers and dataclasses for gpulock."""

from __future__ import annotations

import os
from dataclasses import dataclass


READ_MODE = "read"
WRITE_MODE = "write"


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


@dataclass
class LockConfig:
    poll_ms: int = 200
    timeout_s: int = 1800
    grace_age_s: int = 180
    heartbeat_s: int = 2

    @classmethod
    def from_env(cls) -> "LockConfig":
        defaults = cls()
        return cls(
            poll_ms=env_int("GPULOCK_POLL_MS", defaults.poll_ms),
            timeout_s=env_int("GPULOCK_TIMEOUT_S", defaults.timeout_s),
            grace_age_s=env_int("GPULOCK_GRACE_AGE_S", defaults.grace_age_s),
            heartbeat_s=env_int("GPULOCK_HEARTBEAT_S", defaults.heartbeat_s),
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

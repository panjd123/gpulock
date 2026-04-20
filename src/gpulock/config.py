"""Shared constants, env helpers and dataclasses for gpulock."""

from __future__ import annotations

import os
from dataclasses import dataclass


READ_MODE = "read"
WRITE_MODE = "write"

MODE_ALIAS_MAP = {
    "perf": WRITE_MODE,
    "bench": WRITE_MODE,
    "benchmark": WRITE_MODE,
    "write": WRITE_MODE,
    "check": READ_MODE,
    "correctness": READ_MODE,
    "functional": READ_MODE,
    "test": READ_MODE,
    "read": READ_MODE,
}


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
    poll_ms: int = env_int("GPU_BENCH_LOCK_POLL_MS", 200)
    timeout_s: int = env_int("GPU_BENCH_LOCK_TIMEOUT_S", 1800)
    grace_age_s: int = env_int("GPU_BENCH_LOCK_GRACE_AGE_S", 180)
    heartbeat_s: int = env_int("GPU_BENCH_LOCK_HEARTBEAT_S", 2)
    orphan_check_s: int = env_int("GPU_BENCH_LOCK_ORPHAN_CHECK_S", 5)
    orphan_empty_threshold: int = env_int("GPU_BENCH_LOCK_ORPHAN_EMPTY_THRESHOLD", 6)


@dataclass
class ProbeState:
    last_probe_s: float = 0.0
    last_mtime_ns: int = -1
    last_hb_ms: int = -1
    empty_count: int = 0


@dataclass
class GpuRuntimeState:
    util_gpu: int
    mem_used_mib: int
    mem_total_mib: int
    visible_compute_pids: int
    visible_non_placeholder_pids: int

"""Lock-root resolution and lock metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_lock_root() -> Path:
    """Resolve the directory we use for all gpulock state.

    Order:
        1. ``GPU_BENCH_LOCK_DIR``
        2. ``/var/lock/gpu-benchmark``
        3. ``/tmp/gpu_benchmark_locks``
    """

    env_root = os.getenv("GPU_BENCH_LOCK_DIR", "").strip()
    if env_root:
        path = Path(env_root)
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except PermissionError:
            pass
        return path

    default_root = Path("/var/lock/gpu-benchmark")
    try:
        default_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(default_root, 0o700)
        except PermissionError:
            pass
        return default_root
    except Exception:
        fallback_root = Path("/tmp/gpu_benchmark_locks")
        fallback_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(fallback_root, 0o700)
        except PermissionError:
            pass
        return fallback_root


def read_lock_metadata(lock_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        data = lock_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return meta
    for line in data.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def read_lock_pid(lock_path: Path) -> Optional[int]:
    meta = read_lock_metadata(lock_path)
    raw = meta.get("pid", "")
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def read_last_heartbeat_ms(lock_path: Path) -> int:
    meta = read_lock_metadata(lock_path)
    raw = meta.get("last_heartbeat_ms", "")
    if raw == "":
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1

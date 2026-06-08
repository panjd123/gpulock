"""High-level lock sessions spanning one or more GPUs."""

from __future__ import annotations

import atexit
import os
import signal
from dataclasses import dataclass, field

from .config import LockConfig
from .lock import GpuLock


@dataclass
class MultiGpuLock:
    gpu_ids: list[int]
    mode: str
    config: LockConfig
    skip_gpu_idle_check: bool = False
    idle_streak_s: int = 3
    idle_check_ms: int = 100
    locks: list[GpuLock] = field(default_factory=list, init=False)
    _signals_registered: bool = field(default=False, init=False)
    _atexit_registered: bool = field(default=False, init=False)

    @property
    def gpu_ids_str(self) -> str:
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

    def acquire(self) -> None:
        try:
            for gpu_id in self.gpu_ids:
                lock = GpuLock(
                    gpu_id,
                    mode=self.mode,
                    config=self.config,
                    skip_gpu_idle_check=self.skip_gpu_idle_check,
                    idle_streak_s=self.idle_streak_s,
                    idle_check_ms=self.idle_check_ms,
                    register_signals=False,
                )
                lock.acquire()
                self.locks.append(lock)
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        for lock in reversed(self.locks):
            lock.release()
        self.locks.clear()

    def lock_paths_str(self) -> str:
        return " ".join(str(lock.lock_path) for lock in self.locks if lock.lock_path)

    def child_env_overrides(self) -> dict[str, str]:
        return {
            "GPULOCK_LOCKED_DEVICES": self.gpu_ids_str,
            "GPULOCK_LOCK_MODE": self.mode,
            "CUDA_VISIBLE_DEVICES": self.gpu_ids_str,
        }

    def register_process_cleanup(self) -> None:
        if not self._signals_registered:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
                signal.signal(sig, self._signal_handler)
            self._signals_registered = True
        if not self._atexit_registered:
            atexit.register(self.release)
            self._atexit_registered = True

    def _signal_handler(self, signum, _frame) -> None:
        self.release()
        os._exit(128 + signum)

    def __enter__(self) -> "MultiGpuLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

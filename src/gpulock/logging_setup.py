"""Logging helpers for gpulock CLI/guard."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import env_bool, env_int


def resolve_log_level(default: str = "INFO") -> int:
    raw = os.getenv("GPULOCK_LOG_LEVEL", default).strip().upper()
    return getattr(logging, raw, logging.INFO)


def setup_rotating_logger(
    lock_root: Path,
    name: str,
    filename: str,
    to_stdout: bool,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(resolve_log_level())
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] pid=%(process)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    max_bytes = env_int("GPULOCK_LOG_MAX_BYTES", 20 * 1024 * 1024, minimum=1024)
    backup_count = env_int("GPULOCK_LOG_BACKUP_COUNT", 5, minimum=1)
    log_path = lock_root / filename
    fh = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backup_count)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def setup_guard_logger(lock_root: Path) -> logging.Logger:
    to_stdout = env_bool("GPULOCK_GUARD_LOG_STDOUT", True)
    return setup_rotating_logger(
        lock_root,
        name="gpulock.guard",
        filename="guard.log",
        to_stdout=to_stdout,
    )


def setup_main_logger(lock_root: Path) -> logging.Logger:
    to_stdout = env_bool("GPULOCK_LOG_STDOUT", False)
    return setup_rotating_logger(
        lock_root,
        name="gpulock.main",
        filename="gpulock.log",
        to_stdout=to_stdout,
    )

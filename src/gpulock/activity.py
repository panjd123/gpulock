"""Append-only GPU activity log in a single ``gpu_activity`` table.

Each row is ``(ts, gpu_id, activity_type)`` where ``activity_type`` is
``gpulock`` or ``user_gpu``. Latest rows per GPU/type are read through
``idx_gpu_activity_latest``; rows are never deleted.
"""

from __future__ import annotations

import sqlite3
import time

from .service.common import DEFAULT_IDLE_TIMEOUT

ACTIVITY_TABLE = "gpu_activity"
LATEST_INDEX = "idx_gpu_activity_latest"
ACTIVITY_TYPE_GPULOCK = "gpulock"
ACTIVITY_TYPE_USER_GPU = "user_gpu"

_LEGACY_LAST_TABLES = (
    ("gpu_last_gpulock_activity", ACTIVITY_TYPE_GPULOCK),
    ("gpu_last_user_gpu_activity", ACTIVITY_TYPE_USER_GPU),
    ("gpu_last_activity", ACTIVITY_TYPE_GPULOCK),
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({name})")}


def _create_activity_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {ACTIVITY_TABLE} ("
        "ts REAL NOT NULL, "
        "gpu_id INTEGER NOT NULL, "
        "activity_type TEXT NOT NULL"
        ")"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {LATEST_INDEX} "
        f"ON {ACTIVITY_TABLE}(gpu_id, activity_type, ts DESC)"
    )


def _insert_activity(
    conn: sqlite3.Connection,
    gpu_id: int,
    activity_type: str,
    ts: float,
) -> None:
    conn.execute(
        f"INSERT INTO {ACTIVITY_TABLE}(ts, gpu_id, activity_type) VALUES (?, ?, ?)",
        (ts, gpu_id, activity_type),
    )


def _migrate_legacy_gpu_activity_table(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, ACTIVITY_TABLE):
        return
    columns = _table_columns(conn, ACTIVITY_TABLE)
    if "activity_type" in columns:
        return

    legacy_rows = conn.execute(f"SELECT ts, gpu_id FROM {ACTIVITY_TABLE}").fetchall()
    conn.execute(f"DROP TABLE {ACTIVITY_TABLE}")
    _create_activity_table(conn)
    for ts, gpu_id in legacy_rows:
        _insert_activity(conn, int(gpu_id), ACTIVITY_TYPE_GPULOCK, float(ts))


def _migrate_legacy_last_tables(conn: sqlite3.Connection) -> None:
    for table_name, activity_type in _LEGACY_LAST_TABLES:
        if not _table_exists(conn, table_name):
            continue
        rows = conn.execute(
            f"SELECT gpu_id, last_activity_ts FROM {table_name}",
        ).fetchall()
        for gpu_id, last_ts in rows:
            _insert_activity(conn, int(gpu_id), activity_type, float(last_ts))
        conn.execute(f"DROP TABLE {table_name}")


def init_guard_db(conn: sqlite3.Connection) -> None:
    _migrate_legacy_gpu_activity_table(conn)
    _create_activity_table(conn)
    _migrate_legacy_last_tables(conn)


def record_gpulock_activity(conn: sqlite3.Connection, gpu_id: int, ts: float) -> None:
    _insert_activity(conn, gpu_id, ACTIVITY_TYPE_GPULOCK, ts)


def record_user_gpu_activity(conn: sqlite3.Connection, gpu_id: int, ts: float) -> None:
    _insert_activity(conn, gpu_id, ACTIVITY_TYPE_USER_GPU, ts)


record_gpulock_event = record_gpulock_activity


def _latest_activity_ts(
    conn: sqlite3.Connection,
    gpu_id: int,
    activity_type: str,
) -> float | None:
    row = conn.execute(
        f"SELECT ts FROM {ACTIVITY_TABLE} "
        "WHERE gpu_id=? AND activity_type=? "
        "ORDER BY ts DESC LIMIT 1",
        (gpu_id, activity_type),
    ).fetchone()
    if row is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _activity_age_s(last_ts: float | None, now: float) -> float | None:
    if last_ts is None:
        return None
    return max(now - last_ts, 0.0)


def last_gpulock_activity_age_s(conn: sqlite3.Connection, gpu_id: int, now: float) -> float | None:
    return _activity_age_s(_latest_activity_ts(conn, gpu_id, ACTIVITY_TYPE_GPULOCK), now)


def last_user_gpu_activity_age_s(conn: sqlite3.Connection, gpu_id: int, now: float) -> float | None:
    return _activity_age_s(_latest_activity_ts(conn, gpu_id, ACTIVITY_TYPE_USER_GPU), now)


def _has_recent_ts(last_ts: float | None, window_s: float, now: float | None = None) -> bool:
    if last_ts is None:
        return False
    now_ts = time.time() if now is None else now
    return (now_ts - last_ts) <= max(window_s, 0.0)


def has_recent_gpulock_activity(
    conn: sqlite3.Connection,
    gpu_id: int,
    window_s: float = DEFAULT_IDLE_TIMEOUT,
    *,
    now: float | None = None,
) -> bool:
    return _has_recent_ts(
        _latest_activity_ts(conn, gpu_id, ACTIVITY_TYPE_GPULOCK),
        window_s,
        now,
    )


def has_recent_user_gpu_activity(
    conn: sqlite3.Connection,
    gpu_id: int,
    window_s: float = DEFAULT_IDLE_TIMEOUT,
    *,
    now: float | None = None,
) -> bool:
    return _has_recent_ts(
        _latest_activity_ts(conn, gpu_id, ACTIVITY_TYPE_USER_GPU),
        window_s,
        now,
    )


def has_recent_user_idle_activity(
    conn: sqlite3.Connection,
    gpu_id: int,
    window_s: float = DEFAULT_IDLE_TIMEOUT,
    *,
    now: float | None = None,
) -> bool:
    """True when either gpulock or this user's GPU compute was recent."""
    return has_recent_gpulock_activity(conn, gpu_id, window_s, now=now) or has_recent_user_gpu_activity(
        conn, gpu_id, window_s, now=now
    )

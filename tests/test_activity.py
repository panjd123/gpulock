from __future__ import annotations

import os
import sqlite3
import time

from gpulock import activity, gpu


def test_activity_table_has_latest_index():
    conn = sqlite3.connect(":memory:")
    activity.init_guard_db(conn)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (activity.LATEST_INDEX,),
    ).fetchone()
    assert row is not None


def test_latest_activity_uses_indexed_lookup():
    conn = sqlite3.connect(":memory:")
    activity.init_guard_db(conn)
    now = time.time()
    for offset in range(20):
        activity.record_gpulock_activity(conn, 1, now + offset)
    conn.commit()

    plan = conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT ts FROM gpu_activity WHERE gpu_id=? AND activity_type=? "
        "ORDER BY ts DESC LIMIT 1",
        (1, activity.ACTIVITY_TYPE_GPULOCK),
    ).fetchall()
    plan_text = "\n".join(str(row) for row in plan)
    assert activity.LATEST_INDEX in plan_text or "USING INDEX" in plan_text
    assert activity.last_gpulock_activity_age_s(conn, 1, now + 25.0) == 6.0


def test_activity_tables_track_gpulock_and_user_gpu_separately():
    conn = sqlite3.connect(":memory:")
    activity.init_guard_db(conn)
    now = time.time()
    activity.record_gpulock_event(conn, 7, now)
    activity.record_user_gpu_activity(conn, 7, now + 5.0)
    conn.commit()

    assert activity.has_recent_gpulock_activity(conn, 7, window_s=10.0, now=now + 6.0)
    assert activity.has_recent_user_gpu_activity(conn, 7, window_s=10.0, now=now + 6.0)
    assert activity.has_recent_user_idle_activity(conn, 7, window_s=10.0, now=now + 6.0)

    assert activity.last_gpulock_activity_age_s(conn, 7, now + 6.0) == 6.0
    assert activity.last_user_gpu_activity_age_s(conn, 7, now + 6.0) == 1.0


def test_user_idle_activity_counts_either_source():
    conn = sqlite3.connect(":memory:")
    activity.init_guard_db(conn)
    now = time.time()
    activity.record_user_gpu_activity(conn, 3, now)
    conn.commit()

    assert activity.has_recent_user_idle_activity(conn, 3, window_s=5.0, now=now + 1.0)
    assert not activity.has_recent_gpulock_activity(conn, 3, window_s=5.0, now=now + 10.0)


def test_user_gpu_compute_pids_filters_placeholder_and_other_users(monkeypatch):
    gpu_id = 99
    owner = os.getuid()
    other_uid = owner + 1 if owner < 65000 else owner - 1

    monkeypatch.setattr(gpu, "gpu_compute_pids", lambda _index: {1001, 1002, 1003})
    monkeypatch.setattr(
        gpu,
        "process_owner_uid",
        lambda pid: owner if pid in {1001, 1003} else other_uid,
    )
    monkeypatch.setattr(gpu, "is_placeholder_process", lambda pid: pid == 1003)

    pids = gpu.user_gpu_compute_pids(gpu_id, uid=owner)
    assert pids == {1001}

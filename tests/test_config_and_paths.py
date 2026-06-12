from __future__ import annotations

import os
import sqlite3
import time

from gpulock import config, lock, paths


def test_env_helpers(monkeypatch):
    monkeypatch.setenv("GP_TEST_INT_OK", "42")
    monkeypatch.setenv("GP_TEST_INT_BAD", "bogus")
    monkeypatch.setenv("GP_TEST_INT_LOW", "0")
    monkeypatch.setenv("GP_TEST_BOOL_TRUE", "yes")
    monkeypatch.setenv("GP_TEST_BOOL_FALSE", "no")

    assert config.env_int("GP_TEST_INT_OK", 1) == 42
    assert config.env_int("GP_TEST_INT_BAD", 7) == 7
    assert config.env_int("GP_TEST_INT_LOW", 5, minimum=5) == 5
    assert config.env_int("GP_TEST_NOT_SET_XYZ", 9) == 9
    assert config.env_bool("GP_TEST_BOOL_TRUE", False) is True
    assert config.env_bool("GP_TEST_BOOL_FALSE", True) is False
    assert config.env_bool("GP_TEST_BOOL_NONE_XYZ", True) is True


def test_lock_config_from_env_reads_current_environment(monkeypatch):
    monkeypatch.setenv("GPULOCK_TIMEOUT_S", "33")
    monkeypatch.setenv("GPULOCK_POLL_MS", "0")
    cfg = config.LockConfig.from_env()

    assert cfg.timeout_s == 33
    assert cfg.poll_ms == 1


def test_resolve_lock_root_honors_env(lock_root):
    got = paths.resolve_lock_root()
    assert got == lock_root
    assert oct(got.stat().st_mode & 0o777) == "0o700"


def test_lock_metadata_round_trip(lock_root):
    gpu_dir = lock_root / "gpu99"
    gpu_dir.mkdir(parents=True)
    fake_lock = gpu_dir / "fake.lock"
    fake_lock.write_text(
        "pid=12345\n"
        "lock_mode=write\n"
        "last_heartbeat_ms=987654321\n"
        "stray line without equals\n"
        "extra=value\n",
        encoding="utf-8",
    )

    meta = paths.read_lock_metadata(fake_lock)
    assert meta["pid"] == "12345"
    assert "stray line without equals" not in meta
    assert paths.read_lock_pid(fake_lock) == 12345
    assert paths.read_last_heartbeat_ms(fake_lock) == 987654321


def test_notify_guard_activity_persists_last_and_history(lock_root):
    lock.notify_guard_activity(lock_root, 99, "read")

    assert (lock_root / "gpu99" / "activity.pulse").exists()
    guard_db = lock_root / "guard.db"
    assert guard_db.exists()
    conn = sqlite3.connect(str(guard_db))
    try:
        last_row = conn.execute(
            "SELECT ts FROM gpu_activity WHERE gpu_id=? AND activity_type='gpulock' "
            "ORDER BY ts DESC LIMIT 1",
            (99,),
        ).fetchone()
        hist_row = conn.execute(
            "SELECT COUNT(*) FROM gpu_activity WHERE gpu_id=? AND activity_type='gpulock'",
            (99,),
        ).fetchone()
    finally:
        conn.close()

    assert last_row is not None
    assert hist_row is not None
    assert hist_row[0] >= 1


def test_read_lock_acquire_release(lock_root):
    read_lock = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(timeout_s=5),
    )
    read_lock.acquire()
    assert read_lock.fd is not None
    assert read_lock.lock_path is not None
    assert read_lock.lock_path.parent.name == "readers"
    assert read_lock.lock_path.exists()

    meta_live = paths.read_lock_metadata(read_lock.lock_path)
    assert meta_live["lock_mode"] == "read"
    assert meta_live["device_id"] == "99"
    assert int(meta_live["pid"]) == os.getpid()

    read_lock_b = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(timeout_s=5),
    )
    read_lock_b.acquire()
    readers = sorted((lock_root / "gpu99" / "readers").glob("*.lock"))
    assert len(readers) == 2

    read_lock_b.release()
    read_lock.release()
    assert read_lock_b.fd is None
    assert read_lock.fd is None
    assert sorted((lock_root / "gpu99" / "readers").glob("*.lock")) == []


def test_multi_gpu_locking(lock_root):
    read_a = lock.GpuLock(98, mode="read", config=config.LockConfig(timeout_s=5), register_signals=False)
    read_b = lock.GpuLock(99, mode="read", config=config.LockConfig(timeout_s=5), register_signals=False)
    read_a.acquire()
    read_b.acquire()
    assert read_a.fd is not None
    assert read_b.fd is not None
    read_b.release()
    read_a.release()

    write_a = lock.GpuLock(98, mode="write", config=config.LockConfig(timeout_s=5), register_signals=False)
    write_b = lock.GpuLock(99, mode="write", config=config.LockConfig(timeout_s=5), register_signals=False)
    write_a.acquire()
    write_b.acquire()
    assert (lock_root / "gpu98" / "write.lock").exists()
    assert (lock_root / "gpu99" / "write.lock").exists()
    write_b.release()
    write_a.release()
    assert not (lock_root / "gpu98" / "write.lock").exists()
    assert not (lock_root / "gpu99" / "write.lock").exists()


def test_stale_lock_cleanup_keeps_dead_parent_lock_while_gpu_has_process(lock_root, monkeypatch):
    gpu_lock = lock.GpuLock(
        physical_device_id=99,
        mode="write",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3),
        register_signals=False,
    )
    stale = gpu_lock.writer_path
    stale.write_text(
        "pid=99999999\n"
        "lock_mode=write\n"
        "last_heartbeat_ms=1\n",
        encoding="utf-8",
    )
    old = time.time() - 10
    os.utime(stale, (old, old))

    monkeypatch.setattr(lock, "pid_exists", lambda _pid: False)
    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: True)

    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()

    assert stale.exists()


def test_stale_lock_cleanup_requires_stable_observation_and_empty_gpu(lock_root, monkeypatch):
    gpu_lock = lock.GpuLock(
        physical_device_id=99,
        mode="write",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3),
        register_signals=False,
    )
    stale = gpu_lock.writer_path
    stale.write_text(
        "pid=99999999\n"
        "lock_mode=write\n"
        "last_heartbeat_ms=1\n",
        encoding="utf-8",
    )
    old = time.time() - 10
    os.utime(stale, (old, old))

    monkeypatch.setattr(lock, "pid_exists", lambda _pid: False)
    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: False)

    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()
    assert stale.exists()

    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()
    assert not stale.exists()

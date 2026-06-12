from __future__ import annotations

import os
import time

from gpulock import diagnostics
from gpulock.config import GpuRuntimeState


def _write_reader_lock(lock_root, gpu_id: int, pid: int, cmdline: str) -> None:
    readers = lock_root / f"gpu{gpu_id}" / "readers"
    readers.mkdir(parents=True, exist_ok=True)
    (readers / f"reader-{pid}.lock").write_text(
        f"pid={pid}\n"
        "hostname=testhost\n"
        "device_id=0\n"
        "lock_mode=read\n"
        "start_time=1.0\n"
        f"cmdline={cmdline}\n"
        "last_heartbeat_ms=999\n",
        encoding="utf-8",
    )


def test_build_abnormal_exit_report_lists_peers_and_scheduling_note(lock_root, monkeypatch):
    self_pid = os.getpid()
    peer_pid = self_pid + 1000
    _write_reader_lock(lock_root, 0, self_pid, "gpulock read 0 -- python self.py")
    _write_reader_lock(lock_root, 0, peer_pid, "gpulock read 0 -- python peer.py")

    monkeypatch.setattr(diagnostics, "pid_exists", lambda pid: pid in {self_pid, peer_pid})
    monkeypatch.setattr(
        diagnostics,
        "gpu_compute_memory_mib_by_pid",
        lambda _gpu_id: {peer_pid + 1: 40000},
    )
    monkeypatch.setattr(
        diagnostics,
        "_descendant_pids",
        lambda pid: {pid, peer_pid + 1} if pid == peer_pid else {pid},
    )
    monkeypatch.setattr(
        diagnostics,
        "gpu_runtime_state_by_index",
        lambda _gpu_id: GpuRuntimeState(
            util_gpu=0,
            mem_used_mib=76000,
            mem_total_mib=80000,
            visible_compute_pids=2,
            visible_non_placeholder_pids=1,
        ),
    )
    monkeypatch.setattr(diagnostics, "is_placeholder_process", lambda _pid: False)

    report = diagnostics.build_abnormal_exit_report(
        [0],
        mode="read",
        returncode=1,
        lock_root=lock_root,
        self_pid=self_pid,
    )

    assert "abnormal child exit (rc=1)" in report
    assert "Scheduling: assign GPUs up front" in report
    assert "this_session" in report
    assert "peer_active" in report
    assert "python peer.py" in report
    assert "gpu_mem=40000MiB" in report
    assert "multiple `read`/`check` sessions share this GPU" in report


def test_build_abnormal_exit_report_includes_recently_released_peer(lock_root, monkeypatch):
    self_pid = os.getpid()
    peer_pid = self_pid + 1000
    _write_reader_lock(lock_root, 0, self_pid, "gpulock read 0 -- python self.py")
    diagnostics.record_release_tombstone(
        0,
        mode="read",
        wrapper_pid=peer_pid,
        cmdline="gpulock read 0 -- python peer.py",
        child_rc=1,
        lock_root=lock_root,
        released_at=time.time() - 30.0,
    )

    monkeypatch.setattr(diagnostics, "pid_exists", lambda pid: pid == self_pid)
    monkeypatch.setattr(diagnostics, "gpu_compute_memory_mib_by_pid", lambda _gpu_id: {})
    monkeypatch.setattr(
        diagnostics,
        "gpu_runtime_state_by_index",
        lambda _gpu_id: GpuRuntimeState(
            util_gpu=0,
            mem_used_mib=76000,
            mem_total_mib=80000,
            visible_compute_pids=1,
            visible_non_placeholder_pids=1,
        ),
    )
    monkeypatch.setattr(diagnostics, "is_placeholder_process", lambda _pid: False)

    report = diagnostics.build_abnormal_exit_report(
        [0],
        mode="read",
        returncode=1,
        lock_root=lock_root,
        self_pid=self_pid,
    )

    assert "peer_recent" in report
    assert "python peer.py" in report
    assert "released=30.0s ago" in report
    assert "multiple `read`/`check` sessions share this GPU" in report


def test_should_emit_abnormal_exit_report_honors_disable_flag(monkeypatch):
    assert diagnostics.should_emit_abnormal_exit_report(1) is True
    monkeypatch.setenv("GPULOCK_NO_EXIT_REPORT", "1")
    assert diagnostics.should_emit_abnormal_exit_report(1) is False
    assert diagnostics.should_emit_abnormal_exit_report(0) is False

from __future__ import annotations

import json
import os
import logging
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from gpulock import config, gpu, guard, lock, placeholder
from gpulock.guard import _init_guard_db
from gpulock.config import GpuRuntimeState
from gpulock.service import supervisor
from gpulock.service.common import DEFAULT_PLACEHOLDER_IDLE_S, guard_status_path


def _write_stale_writer(lock_obj: lock.GpuLock, pid: int = 99999999) -> None:
    stale = lock_obj.writer_path
    stale.write_text(
        f"pid={pid}\n"
        "lock_mode=write\n"
        "last_heartbeat_ms=1\n",
        encoding="utf-8",
    )
    old = time.time() - 10
    os.utime(stale, (old, old))


def test_stale_lock_cleanup_keeps_live_pid_lock(lock_root, monkeypatch):
    gpu_lock = lock.GpuLock(
        physical_device_id=99,
        mode="write",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3),
        register_signals=False,
    )
    _write_stale_writer(gpu_lock, pid=12345)

    monkeypatch.setattr(lock, "pid_exists", lambda pid: pid == 12345)
    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: False)

    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()

    assert gpu_lock.writer_path.exists()
    assert gpu_lock._stale_probe == {}


def test_stale_lock_cleanup_missing_pid_uses_grace_age(lock_root, monkeypatch):
    gpu_lock = lock.GpuLock(
        physical_device_id=99,
        mode="write",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=30),
        register_signals=False,
    )
    stale = gpu_lock.writer_path
    stale.write_text(
        "lock_mode=write\n"
        "last_heartbeat_ms=1\n",
        encoding="utf-8",
    )
    too_young_for_missing_pid = time.time() - 10
    os.utime(stale, (too_young_for_missing_pid, too_young_for_missing_pid))

    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: False)

    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()
        gpu_lock._cleanup_stale_locks_locked()

    assert stale.exists()

    old_enough_for_missing_pid = time.time() - 40
    os.utime(stale, (old_enough_for_missing_pid, old_enough_for_missing_pid))
    with gpu_lock._state_gate():
        gpu_lock._cleanup_stale_locks_locked()
        gpu_lock._cleanup_stale_locks_locked()

    assert not stale.exists()


def test_acquire_parks_placeholder_before_cleaning_stale_writer(lock_root, monkeypatch):
    probe_lock = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3, timeout_s=2, poll_ms=10),
        register_signals=False,
    )
    _write_stale_writer(probe_lock)

    calls: list[str] = []
    monkeypatch.setattr(lock, "notify_guard_activity", lambda *_args, **_kwargs: calls.append("notify"))
    monkeypatch.setattr(lock, "park_placeholder", lambda *_args, **_kwargs: calls.append("park") or True)
    monkeypatch.setattr(lock, "kill_placeholder", lambda *_args, **_kwargs: calls.append("kill_placeholder"))
    monkeypatch.setattr(
        lock,
        "kill_visible_placeholder_compute_pids",
        lambda *_args, **_kwargs: calls.append("kill_visible_placeholder_compute_pids"),
    )
    monkeypatch.setattr(lock, "pid_exists", lambda pid: pid == os.getpid())
    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: False)

    acquired = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3, timeout_s=2, poll_ms=10),
        register_signals=False,
    )
    acquired.acquire()
    try:
        assert calls[:2] == ["notify", "park"]
        assert "kill_placeholder" not in calls
        assert not probe_lock.writer_path.exists()
        assert acquired.lock_path is not None
        assert acquired.lock_path.exists()
    finally:
        acquired.release()


def test_acquire_kills_unresponsive_placeholder_then_cleans_stale_writer(lock_root, monkeypatch):
    probe_lock = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3, timeout_s=2, poll_ms=10),
        register_signals=False,
    )
    _write_stale_writer(probe_lock)

    calls: list[str] = []
    monkeypatch.setattr(lock, "notify_guard_activity", lambda *_args, **_kwargs: calls.append("notify"))
    def fail_to_park_placeholder(*_args, **_kwargs):
        calls.append("park")
        return False

    monkeypatch.setattr(lock, "park_placeholder", fail_to_park_placeholder)
    monkeypatch.setattr(lock, "kill_placeholder", lambda *_args, **_kwargs: calls.append("kill_placeholder"))
    monkeypatch.setattr(
        lock,
        "kill_visible_placeholder_compute_pids",
        lambda *_args, **_kwargs: calls.append("kill_visible_placeholder_compute_pids"),
    )
    monkeypatch.setattr(lock, "pid_exists", lambda pid: pid == os.getpid())
    monkeypatch.setattr(lock, "gpu_has_processes_by_index", lambda _gpu_id: False)

    acquired = lock.GpuLock(
        physical_device_id=99,
        mode="read",
        config=config.LockConfig(heartbeat_s=1, grace_age_s=3, timeout_s=2, poll_ms=10),
        register_signals=False,
    )
    acquired.acquire()
    try:
        assert calls[:3] == ["notify", "park", "kill_placeholder"]
        assert "kill_visible_placeholder_compute_pids" in calls
        assert not probe_lock.writer_path.exists()
        assert acquired.lock_path is not None
    finally:
        acquired.release()


def test_gpu_busy_reason_ignores_placeholder_only_compute_process(monkeypatch):
    monkeypatch.setattr(
        gpu,
        "gpu_runtime_state_by_index",
        lambda _gpu_id: GpuRuntimeState(
            util_gpu=92,
            mem_used_mib=40000,
            mem_total_mib=80000,
            visible_compute_pids=1,
            visible_non_placeholder_pids=0,
        ),
    )

    busy, reason = gpu.gpu_busy_reason_for_perf(0)

    assert busy is False
    assert "idle_by=placeholder_only" in reason


def test_gpu_busy_reason_blocks_non_placeholder_process_with_util(monkeypatch):
    monkeypatch.setattr(
        gpu,
        "gpu_runtime_state_by_index",
        lambda _gpu_id: GpuRuntimeState(
            util_gpu=12,
            mem_used_mib=40000,
            mem_total_mib=80000,
            visible_compute_pids=2,
            visible_non_placeholder_pids=1,
        ),
    )

    busy, reason = gpu.gpu_busy_reason_for_perf(0)

    assert busy is True
    assert "busy_by=util=12%" in reason


def test_placeholder_client_helpers_parse_status_and_missing_socket(lock_root, monkeypatch):
    gpu_dir = lock_root / "gpu99"
    gpu_dir.mkdir()

    assert placeholder.placeholder_command(gpu_dir, "status") == (False, "missing socket")

    monkeypatch.setattr(placeholder, "placeholder_command", lambda *_args, **_kwargs: (True, "ok state=active"))
    assert placeholder.placeholder_state(gpu_dir) == "active"


def test_gpu_has_our_activity_ignores_dead_locks_and_placeholder_pid(lock_root, monkeypatch):
    gpu_dir = lock_root / "gpu99"
    readers = gpu_dir / "readers"
    readers.mkdir(parents=True)
    (gpu_dir / "placeholder.pid").write_text("12345\n", encoding="utf-8")
    (readers / "reader-dead.lock").write_text("pid=99999999\n", encoding="utf-8")

    monkeypatch.setattr(lock, "pid_exists", lambda _pid: False)

    assert lock.gpu_has_our_activity(lock_root, 99) is False

    live_lock = readers / "reader-live.lock"
    live_lock.write_text("pid=12345\n", encoding="utf-8")
    monkeypatch.setattr(lock, "pid_exists", lambda pid: pid == 12345)

    assert lock.gpu_has_our_activity(lock_root, 99) is True


def _wait_until(deadline_s: float, predicate):
    last = None
    while time.monotonic() < deadline_s:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    return last


def _guard_test_env(lock_root: Path, fake_modules: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GPULOCK_LOCK_DIR"] = str(lock_root)
    env["PYTHONPATH"] = f"{fake_modules}:{Path(__file__).resolve().parent.parent / 'src'}"
    return env


def _start_guard_proc(
    env: dict[str, str],
    gpu_id: int,
    *,
    idle_timeout_s: float,
    placeholder_idle_s: float = 0.05,
    guard_poll_s: float = 0.1,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpulock",
            "guard",
            str(gpu_id),
            "--placeholder-idle-s",
            str(placeholder_idle_s),
            "--idle-timeout",
            str(idle_timeout_s),
            "--guard-poll-s",
            str(guard_poll_s),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _write_fake_nvidia_smi(bin_dir: Path, gpu_id: int) -> None:
    script = bin_dir / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        f'GPU_ID="{gpu_id}"\n'
        'UUID="GPU-TEST-UUID-${GPU_ID}"\n'
        'case "$*" in\n'
        '  *query-compute-apps=gpu_uuid,pid*)\n'
        '    PID_FILE="${GPULOCK_LOCK_DIR}/gpu${GPU_ID}/test_compute.pid"\n'
        '    if [ -f "${PID_FILE}" ]; then\n'
        '      COMPUTE_PID="$(cat "${PID_FILE}")"\n'
        '      if kill -0 "${COMPUTE_PID}" 2>/dev/null; then\n'
        '        echo "${UUID},${COMPUTE_PID}"\n'
        "      fi\n"
        "    fi\n"
        "    ;;\n"
        '  *query-gpu=index,uuid*)\n'
        '    echo "${GPU_ID},${UUID}"\n'
        "    ;;\n"
        '  *query-gpu=index,utilization.gpu,memory.used,memory.total*)\n'
        '    echo "${GPU_ID},0,1000,80000"\n'
        "    ;;\n"
        '  *query-gpu=index*)\n'
        '    echo "${GPU_ID}"\n'
        "    ;;\n"
        "  *)\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _gpu_guard_status(lock_root: Path, gpu_id: int) -> dict | None:
    status_path = guard_status_path(lock_root)
    if not status_path.exists():
        return None
    data = json.loads(status_path.read_text(encoding="utf-8"))
    for item in data.get("gpus", []):
        if item.get("gpu_id") == gpu_id:
            return item
    return None


def _assert_guard_alive(guard_proc: subprocess.Popen[str]) -> None:
    if guard_proc.poll() is not None:
        stdout, stderr = guard_proc.communicate(timeout=1)
        raise AssertionError(
            f"guard exited early rc={guard_proc.returncode}\nstdout={stdout}\nstderr={stderr}"
        )


def _stop_process(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _write_fake_placeholder_deps(module_dir: Path) -> None:
    module_dir.mkdir()
    (module_dir / "setproctitle.py").write_text(
        "def setproctitle(_title):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (module_dir / "torch.py").write_text(
        "class _Context:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, *_args):\n"
        "        return False\n"
        "\n"
        "class _Event:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n"
        "    def record(self, *_args, **_kwargs):\n"
        "        return None\n"
        "    def synchronize(self):\n"
        "        return None\n"
        "    def elapsed_time(self, _other):\n"
        "        return 1.0\n"
        "\n"
        "class _Stream:\n"
        "    def wait_stream(self, _stream):\n"
        "        return None\n"
        "    def synchronize(self):\n"
        "        return None\n"
        "\n"
        "class _Graph:\n"
        "    def replay(self):\n"
        "        return None\n"
        "\n"
        "class _Props:\n"
        "    total_memory = 64 * 1024 * 1024\n"
        "\n"
        "class _Cuda:\n"
        "    def set_device(self, _device):\n"
        "        return None\n"
        "    def get_device_properties(self, _device):\n"
        "        return _Props()\n"
        "    def mem_get_info(self):\n"
        "        return (32 * 1024 * 1024, 64 * 1024 * 1024)\n"
        "    def synchronize(self):\n"
        "        return None\n"
        "    def empty_cache(self):\n"
        "        return None\n"
        "    def Event(self, *args, **kwargs):\n"
        "        return _Event(*args, **kwargs)\n"
        "    def Stream(self):\n"
        "        return _Stream()\n"
        "    def current_stream(self):\n"
        "        return _Stream()\n"
        "    def stream(self, _stream):\n"
        "        return _Context()\n"
        "    def CUDAGraph(self):\n"
        "        return _Graph()\n"
        "    def graph(self, *_args, **_kwargs):\n"
        "        return _Context()\n"
        "\n"
        "cuda = _Cuda()\n"
        "float32 = object()\n"
        "float16 = object()\n"
        "\n"
        "def empty(*_args, **_kwargs):\n"
        "    return object()\n"
        "\n"
        "def randn(*_args, **_kwargs):\n"
        "    return object()\n"
        "\n"
        "def mm(*_args, **_kwargs):\n"
        "    return object()\n",
        encoding="utf-8",
    )


def test_wait_placeholder_process_ready_fails_fast_when_worker_exits(lock_root, monkeypatch):
    gpu_dir = lock_root / "gpu99"
    gpu_dir.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    calls = 0

    def fake_placeholder_command(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (False, "missing socket")

    monkeypatch.setattr(guard, "placeholder_command", fake_placeholder_command)
    start = time.monotonic()

    ready = guard._wait_placeholder_process_ready(gpu_dir, proc, timeout_s=30.0)

    assert ready is False
    assert time.monotonic() - start < 2.0
    assert proc.wait(timeout=2) == 7
    assert calls >= 0


def test_guard_exits_after_consecutive_placeholder_start_failures(lock_root, tmp_path, monkeypatch):
    logger = logging.getLogger("gpulock.guard")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "setproctitle.py").write_text(
        "def setproctitle(_title):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (fake_modules / "torch.py").write_text(
        "raise ImportError('libtorch_cuda.so: undefined symbol: ncclCommResume')\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PYTHONPATH", f"{fake_modules}:{Path(__file__).resolve().parent.parent / 'src'}")
    monkeypatch.setattr(guard, "PLACEHOLDER_START_TIMEOUT_S", 1.0)
    monkeypatch.setattr(guard, "PLACEHOLDER_START_FAILURE_EXIT_THRESHOLD", 2)

    rc = guard.cmd_guard(["98", "99", "--placeholder-idle-s", "0.01", "--idle-timeout", "30"])

    assert rc == 70
    log_text = (lock_root / "guard.log").read_text(encoding="utf-8")
    assert "undefined symbol: ncclCommResume" in log_text
    assert "exiting guard for supervisor restart" in log_text


def _run_repeated_gpulock_commands(env: dict[str, str], gpu_id: int, count: int) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    ends: list[float] = []
    for _ in range(count):
        starts.append(time.time())
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpulock",
                "read",
                str(gpu_id),
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        ends.append(time.time())
        assert proc.returncode == 0, proc.stderr
    return starts, ends


def _activity_timestamps(lock_root: Path, gpu_id: int) -> list[float]:
    conn = sqlite3.connect(str(lock_root / "guard.db"))
    try:
        return [
            float(row[0])
            for row in conn.execute(
                "SELECT ts FROM gpu_activity WHERE gpu_id=? AND activity_type='gpulock' ORDER BY ts",
                (gpu_id,),
            )
        ]
    finally:
        conn.close()


def test_default_placeholder_idle_exceeds_repeated_gpulock_start_gap(lock_root):
    env = os.environ.copy()
    env["GPULOCK_LOCK_DIR"] = str(lock_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    conn = _init_guard_db(lock_root)
    conn.close()

    starts, ends = _run_repeated_gpulock_commands(env, gpu_id=99, count=8)
    activity_ts = _activity_timestamps(lock_root, 99)

    start_to_start_gaps = [activity_ts[i + 1] - activity_ts[i] for i in range(len(activity_ts) - 1)]
    launcher_gaps = [starts[i + 1] - ends[i] for i in range(len(ends) - 1)]

    assert len(activity_ts) == 8
    assert max(start_to_start_gaps) < DEFAULT_PLACEHOLDER_IDLE_S
    assert max(launcher_gaps) < 0.02


def test_guard_placeholder_reactivates_after_gpulock_command(lock_root, tmp_path):
    fake_modules = tmp_path / "fake-modules"
    _write_fake_placeholder_deps(fake_modules)

    env = os.environ.copy()
    env["GPULOCK_LOCK_DIR"] = str(lock_root)
    env["PYTHONPATH"] = f"{fake_modules}:{Path(__file__).resolve().parent.parent / 'src'}"

    guard_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpulock",
            "guard",
            "99",
            "--placeholder-idle-s",
            "0.05",
            "--idle-timeout",
            "30",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gpu_dir = lock_root / "gpu99"
    try:
        initial_state = _wait_until(
            time.monotonic() + 10.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        if guard_proc.poll() is not None:
            stdout, stderr = guard_proc.communicate(timeout=1)
            raise AssertionError(f"guard exited early rc={guard_proc.returncode}\nstdout={stdout}\nstderr={stderr}")
        assert initial_state is True

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gpulock",
                "read",
                "99",
                "--",
                sys.executable,
                "-c",
                "import time; print('child-ok', flush=True); time.sleep(1.5)",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        parked_state = _wait_until(
            time.monotonic() + 2.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "parked",
        )
        assert parked_state is True

        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, stderr
        assert "child-ok" in stdout

        reactivated_state = _wait_until(
            time.monotonic() + 6.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        assert reactivated_state is True
    finally:
        _stop_process(guard_proc)


def test_default_placeholder_idle_avoids_reactivation_between_repeated_gpulock_commands(lock_root, tmp_path):
    fake_modules = tmp_path / "fake-modules"
    _write_fake_placeholder_deps(fake_modules)

    env = os.environ.copy()
    env["GPULOCK_LOCK_DIR"] = str(lock_root)
    env["PYTHONPATH"] = f"{fake_modules}:{Path(__file__).resolve().parent.parent / 'src'}"

    guard_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpulock",
            "guard",
            "99",
            "--placeholder-idle-s",
            str(DEFAULT_PLACEHOLDER_IDLE_S),
            "--idle-timeout",
            "30",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gpu_dir = lock_root / "gpu99"
    try:
        initial_state = _wait_until(
            time.monotonic() + 10.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        assert initial_state is True

        _run_repeated_gpulock_commands(env, gpu_id=99, count=5)

        assert placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "parked"

        reactivated_state = _wait_until(
            time.monotonic() + 3.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        assert reactivated_state is True
    finally:
        _stop_process(guard_proc)


def test_idle_timeout_enters_dormant_reflected_in_status_and_reactivates_on_gpulock(
    lock_root,
    tmp_path,
    capsys,
):
    idle_timeout_s = 1
    gpu_id = 99
    fake_modules = tmp_path / "fake-modules"
    _write_fake_placeholder_deps(fake_modules)
    env = _guard_test_env(lock_root, fake_modules)
    guard_proc = _start_guard_proc(env, gpu_id, idle_timeout_s=idle_timeout_s)
    gpu_dir = lock_root / f"gpu{gpu_id}"

    try:
        initial_state = _wait_until(
            time.monotonic() + 10.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        _assert_guard_alive(guard_proc)
        assert initial_state is True

        dormant_snap = _wait_until(
            time.monotonic() + idle_timeout_s + 5.0,
            lambda: snap
            if (snap := _gpu_guard_status(lock_root, gpu_id)) is not None and snap.get("dormant") is True
            else None,
        )
        _assert_guard_alive(guard_proc)
        assert dormant_snap is not None
        assert dormant_snap["placeholder"] == "parked"
        assert dormant_snap["last_gpulock_activity_age_s"] >= idle_timeout_s
        assert "dormant" in str(dormant_snap["reason"])

        status_data = json.loads(guard_status_path(lock_root).read_text(encoding="utf-8"))
        assert status_data["idle_timeout"] == idle_timeout_s
        assert status_data["guard_poll_s"] == 0.1

        supervisor._print_guard_snapshot()
        captured = capsys.readouterr()
        assert f"gpu{gpu_id}: placeholder=parked" in captured.out
        assert "last_gpulock_activity=" in captured.out
        assert "last_user_gpu_activity=" in captured.out
        assert "dormant" in captured.out

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpulock",
                "read",
                str(gpu_id),
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr

        reactivated_state = _wait_until(
            time.monotonic() + 5.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        _assert_guard_alive(guard_proc)
        assert reactivated_state is True

        awake_snap = _wait_until(
            time.monotonic() + 5.0,
            lambda: snap
            if (snap := _gpu_guard_status(lock_root, gpu_id)) is not None
            and snap.get("dormant") is False
            and snap.get("placeholder") == "active"
            and snap.get("last_gpulock_activity_age_s", idle_timeout_s) < idle_timeout_s
            else None,
        )
        assert awake_snap is not None
        assert "dormant" not in str(awake_snap["reason"])
    finally:
        _stop_process(guard_proc)


def test_user_gpu_activity_prevents_dormant_and_is_recorded_in_status(lock_root, tmp_path, capsys):
    idle_timeout_s = 1
    gpu_id = 99
    fake_modules = tmp_path / "fake-modules"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_placeholder_deps(fake_modules)
    _write_fake_nvidia_smi(fake_bin, gpu_id)

    env = _guard_test_env(lock_root, fake_modules)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    guard_proc = _start_guard_proc(env, gpu_id, idle_timeout_s=idle_timeout_s, guard_poll_s=0.1)
    gpu_dir = lock_root / f"gpu{gpu_id}"
    compute_proc: subprocess.Popen[bytes] | None = None
    try:
        initial_state = _wait_until(
            time.monotonic() + 10.0,
            lambda: placeholder.placeholder_state(gpu_dir, timeout_s=0.2) == "active",
        )
        _assert_guard_alive(guard_proc)
        assert initial_state is True

        compute_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        gpu_dir.mkdir(parents=True, exist_ok=True)
        (gpu_dir / "test_compute.pid").write_text(str(compute_proc.pid), encoding="utf-8")

        time.sleep(idle_timeout_s + 0.5)
        snap = _gpu_guard_status(lock_root, gpu_id)
        _assert_guard_alive(guard_proc)
        assert snap is not None
        assert snap.get("dormant") is False
        assert snap.get("last_user_gpu_activity_age_s", 999) < idle_timeout_s
        assert compute_proc.pid in snap.get("user_gpu_compute_pids", [])

        supervisor._print_guard_snapshot()
        captured = capsys.readouterr()
        assert "last_user_gpu_activity=" in captured.out

        (gpu_dir / "test_compute.pid").unlink(missing_ok=True)
        compute_proc.terminate()
        compute_proc.wait(timeout=10)

        dormant_snap = _wait_until(
            time.monotonic() + idle_timeout_s + 5.0,
            lambda: snap
            if (snap := _gpu_guard_status(lock_root, gpu_id)) is not None and snap.get("dormant") is True
            else None,
        )
        _assert_guard_alive(guard_proc)
        assert dormant_snap is not None
    finally:
        (gpu_dir / "test_compute.pid").unlink(missing_ok=True)
        if compute_proc is not None and compute_proc.poll() is None:
            compute_proc.kill()
            compute_proc.wait(timeout=10)
        _stop_process(guard_proc)

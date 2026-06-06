from __future__ import annotations

import json
import os
import time

import pytest

from gpulock.service import common, supervisor
from gpulock.service.common import DEFAULT_IDLE_TIMEOUT, DEFAULT_PLACEHOLDER_IDLE_S


def test_guard_service_config_round_trip(lock_root):
    cfg = common.GuardServiceConfig(
        gpu_ids=[0, 2, 5],
        idle_timeout=900,
        placeholder_idle_s=1.5,
        placeholder_load=False,
        extra_env={"FOO": "bar", "K": "v=eq"},
        python_executable="/usr/bin/python3",
        gpulock_executable="/opt/gpulock/bin/gpulock",
    )

    saved = cfg.save()
    assert saved.exists()
    assert oct(saved.stat().st_mode & 0o777) == "0o600"
    assert common.GuardServiceConfig.load() == cfg
    assert cfg.to_guard_argv() == [
        "guard",
        "0",
        "2",
        "5",
        "--idle-timeout",
        "900",
        "--placeholder-idle-s",
        "1.5",
        "--no-placeholder-load",
    ]


def test_supervisord_conf_rendering(lock_root):
    cfg = common.GuardServiceConfig(
        gpu_ids=[0, 2, 5],
        idle_timeout=900,
        placeholder_idle_s=1.5,
        placeholder_load=False,
        extra_env={"FOO": "bar", "K": "v=eq"},
        python_executable="/usr/bin/python3",
        gpulock_executable="/opt/gpulock/bin/gpulock",
    )

    conf_text = supervisor.render_conf(cfg)
    for needle in (
        "[program:gpulock-guard]",
        "[supervisord]",
        "[unix_http_server]",
        "/opt/gpulock/bin/gpulock guard 0 2 5 --idle-timeout 900",
        'environment=FOO="bar",K="v=eq"',
        "autorestart=true",
        "stdout_logfile=%(here)s/guard.log",
    ):
        assert needle in conf_text


def test_service_install_status_config_uninstall(run_cli, lock_root):
    service_dir = lock_root / "service"

    proc = run_cli(["service", "status"])
    assert proc.returncode == 4
    assert "installed:" in proc.stdout
    assert "no" in proc.stdout

    proc = run_cli([
        "service",
        "install",
        "--no-start",
        "--gpu-ids",
        "0,1",
        "--idle-timeout",
        "600",
        "--no-placeholder-load",
        "--env",
        "FOO=bar",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "config saved to" in proc.stdout
    assert (service_dir / "config.json").exists()
    assert (service_dir / "supervisord.conf").exists()

    saved_cfg = json.loads((service_dir / "config.json").read_text())
    assert saved_cfg["gpu_ids"] == [0, 1]
    assert saved_cfg["idle_timeout"] == 600
    assert saved_cfg["placeholder_idle_s"] == DEFAULT_PLACEHOLDER_IDLE_S
    assert saved_cfg["placeholder_load"] is False
    assert saved_cfg["extra_env"]["FOO"] == "bar"

    conf_on_disk = (service_dir / "supervisord.conf").read_text()
    for needle in (
        "[program:gpulock-guard]",
        "--idle-timeout 600",
        "--no-placeholder-load",
        'environment=FOO="bar"',
    ):
        assert needle in conf_on_disk

    proc = run_cli(["service", "status"])
    assert proc.returncode == 3
    assert "installed:    yes" in proc.stdout
    assert "supervisord:  stopped" in proc.stdout

    proc = run_cli(["service", "config", "path"])
    assert proc.returncode == 0
    assert proc.stdout.strip().endswith("config.json")

    proc = run_cli(["service", "config", "show"])
    assert proc.returncode == 0
    assert "gpu_ids=0,1" in proc.stdout
    assert "idle_timeout=600" in proc.stdout
    assert f"placeholder_idle_s={DEFAULT_PLACEHOLDER_IDLE_S}" in proc.stdout
    assert "placeholder_load=false" in proc.stdout

    proc = run_cli(["service", "config", "get", "idle_timeout"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "600"

    proc = run_cli(["service", "config", "get", "bogus"])
    assert proc.returncode != 0

    proc = run_cli([
        "service",
        "config",
        "set",
        "gpu_ids=2,3,4",
        "idle_timeout=1234",
        "placeholder_load=true",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "service restart" in proc.stdout
    saved_cfg2 = json.loads((service_dir / "config.json").read_text())
    assert saved_cfg2["gpu_ids"] == [2, 3, 4]
    assert saved_cfg2["idle_timeout"] == 1234
    assert saved_cfg2["placeholder_load"] is True

    proc = run_cli(["service", "config", "set", "idle_timeout=not-a-number"])
    assert proc.returncode != 0

    proc = run_cli(["service", "config", "unset", "idle_timeout"])
    assert proc.returncode == 0
    saved_cfg3 = json.loads((service_dir / "config.json").read_text())
    assert saved_cfg3["idle_timeout"] == DEFAULT_IDLE_TIMEOUT

    proc = run_cli(["service", "config", "set", "placeholder_idle_s=2.5"])
    assert proc.returncode == 0
    proc = run_cli(["service", "config", "unset", "placeholder_idle_s"])
    assert proc.returncode == 0
    saved_cfg4 = json.loads((service_dir / "config.json").read_text())
    assert saved_cfg4["placeholder_idle_s"] == DEFAULT_PLACEHOLDER_IDLE_S

    proc = run_cli(["service", "uninstall"])
    assert proc.returncode == 0, proc.stderr
    assert not (service_dir / "config.json").exists()
    assert not (service_dir / "supervisord.conf").exists()


def test_supervisord_daemon_lifecycle(run_cli, lock_root):
    ok, sup_err = supervisor.supervisor_available()
    if not ok:
        pytest.skip(f"supervisor package not importable: {sup_err}")

    service_dir = lock_root / "service"
    proc = run_cli([
        "service",
        "install",
        "--no-start",
        "--gpu-ids",
        "0",
        "--idle-timeout",
        "10",
        "--no-placeholder-load",
    ])
    assert proc.returncode == 0, proc.stderr

    proc = run_cli(["service", "start"])
    assert proc.returncode == 0, proc.stderr

    sup_pid_path = service_dir / "supervisord.pid"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not sup_pid_path.exists():
        time.sleep(0.05)
    assert sup_pid_path.exists()

    sup_pid = int(sup_pid_path.read_text().strip())
    assert sup_pid > 0

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    assert alive(sup_pid)
    time.sleep(0.5)
    assert alive(sup_pid)

    guard_log = service_dir / "guard.log"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not guard_log.exists():
        time.sleep(0.1)
    assert guard_log.exists()

    proc = run_cli(["service", "status"])
    assert "supervisord:  running" in proc.stdout

    proc = run_cli(["service", "logs", "-n", "20"])
    assert proc.returncode == 0

    proc = run_cli(["service", "stop"], timeout=40)
    assert proc.returncode == 0, proc.stderr

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and alive(sup_pid):
        time.sleep(0.1)
    assert not alive(sup_pid)
    assert not sup_pid_path.exists()

    proc = run_cli(["service", "stop"])
    assert proc.returncode == 0
    proc = run_cli(["service", "uninstall"])
    assert proc.returncode == 0
    assert not (service_dir / "config.json").exists()
    assert not (service_dir / "supervisord.conf").exists()

"""End-to-end functional smoke test for the gpulock refactor.

Runs entirely in user-space; uses isolated GPU_BENCH_LOCK_DIR so it can run on
a shared host without disturbing real state. Exercises:

* module import graph
* env helpers
* GuardServiceConfig save/load round-trip
* lock acquire+release for read mode (write mode skipped: needs nvidia driver)
* lock metadata persisted on disk
* gpulock service install --no-start / status / uninstall lifecycle
* gpulock service config show/get/set/unset
* install.sh sanity (no args)
* supervisord daemon lifecycle (real fork)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/tmp/gpulock-tests/scratch")
LOCK_ROOT = ROOT / "lock-root"
shutil.rmtree(ROOT, ignore_errors=True)
LOCK_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["GPU_BENCH_LOCK_DIR"] = str(LOCK_ROOT)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    status = "PASS" if ok else "FAIL"
    detail_s = str(detail) if detail else ""
    print(f"  [{status}] {name}{(' :: ' + detail_s) if detail_s else ''}")
    if not ok:
        failures.append(f"{name} :: {detail_s}" if detail_s else name)


print("=== module imports ===")
try:
    import gpulock  # noqa: F401
    from gpulock import cli, guard, lock, gpu, config, paths, placeholder, logging_setup  # noqa: F401
    from gpulock.service import cli as service_cli, common, supervisor  # noqa: F401
    check("import gpulock + all submodules", True)
except Exception as e:  # pragma: no cover
    check("import gpulock + all submodules", False, repr(e))
    raise

print("\n=== config.env_int / env_bool ===")
os.environ["GP_TEST_INT_OK"] = "42"
os.environ["GP_TEST_INT_BAD"] = "bogus"
os.environ["GP_TEST_INT_LOW"] = "0"
check("env_int: parses int", config.env_int("GP_TEST_INT_OK", 1) == 42)
check("env_int: defaults on bad", config.env_int("GP_TEST_INT_BAD", 7) == 7)
check("env_int: enforces minimum", config.env_int("GP_TEST_INT_LOW", 5, minimum=5) == 5)
check("env_int: missing -> default", config.env_int("GP_TEST_NOT_SET_XYZ", 9) == 9)
os.environ["GP_TEST_BOOL_TRUE"] = "yes"
os.environ["GP_TEST_BOOL_FALSE"] = "no"
check("env_bool: yes -> True", config.env_bool("GP_TEST_BOOL_TRUE", False) is True)
check("env_bool: no -> False", config.env_bool("GP_TEST_BOOL_FALSE", True) is False)
check("env_bool: missing -> default", config.env_bool("GP_TEST_BOOL_NONE_XYZ", True) is True)


print("\n=== resolve_lock_root ===")
got = paths.resolve_lock_root()
check("resolve_lock_root honors GPU_BENCH_LOCK_DIR", got == LOCK_ROOT, str(got))
check("lock_root mode bits = 0o700", oct(got.stat().st_mode & 0o777) == "0o700")


print("\n=== lock metadata round-trip ===")
gpu_dir = LOCK_ROOT / "gpu99"
gpu_dir.mkdir(parents=True, exist_ok=True)
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
check("read_lock_metadata: parses key=value", meta.get("pid") == "12345")
check("read_lock_metadata: ignores garbage", "stray line without equals" not in meta)
check("read_lock_pid", paths.read_lock_pid(fake_lock) == 12345)
check("read_last_heartbeat_ms", paths.read_last_heartbeat_ms(fake_lock) == 987654321)


print("\n=== GuardServiceConfig round-trip ===")
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
check("config.save writes file", saved.exists())
check("config.save chmod 0o600", oct(saved.stat().st_mode & 0o777) == "0o600")
loaded = common.GuardServiceConfig.load()
check("config round-trip equal", loaded == cfg, f"loaded={loaded}")
argv = cfg.to_guard_argv()
check(
    "to_guard_argv shape",
    argv == ["guard", "0", "2", "5", "--idle-timeout", "900",
             "--placeholder-idle-s", "1.5", "--no-placeholder-load"],
    str(argv),
)


print("\n=== supervisord conf rendering ===")
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
    check(f"rendered conf contains {needle!r}", needle in conf_text, conf_text[:200])


print("\n=== read lock acquire+release (no nvidia driver needed) ===")
read_lock = lock.GpuBenchLock(physical_device_id=99, mode="read",
                              config=config.LockConfig(timeout_s=5))
read_lock.acquire()
check("read lock acquired", read_lock.fd is not None and read_lock.lock_path is not None)
check(
    "read lock file exists under readers/",
    read_lock.lock_path.parent.name == "readers" and read_lock.lock_path.exists(),
)
meta_live = paths.read_lock_metadata(read_lock.lock_path)
check(
    "read lock metadata populated",
    meta_live.get("lock_mode") == "read" and meta_live.get("device_id") == "99"
    and int(meta_live.get("pid", "-1")) == os.getpid(),
    str(meta_live)[:200],
)

read_lock_b = lock.GpuBenchLock(physical_device_id=99, mode="read",
                                config=config.LockConfig(timeout_s=5))
read_lock_b.acquire()
readers = sorted((LOCK_ROOT / "gpu99" / "readers").glob("*.lock"))
check("two concurrent read locks present", len(readers) == 2, f"readers={readers}")
read_lock_b.release()
check("read lock B released cleanly", read_lock_b.fd is None and read_lock_b.lock_path is None)
read_lock.release()
check("read lock A released cleanly", read_lock.fd is None and read_lock.lock_path is None)
remaining = sorted((LOCK_ROOT / "gpu99" / "readers").glob("*.lock"))
check("readers dir cleaned after release", remaining == [], f"remaining={remaining}")


print("\n=== gpulock service install --no-start / status / uninstall (no daemon) ===")
env = os.environ.copy()
env["GPU_BENCH_LOCK_DIR"] = str(LOCK_ROOT)
env["PYTHONPATH"] = str(REPO / "src")


def run_cli(args: list[str], timeout: int = 20) -> tuple[int, str, str]:
    p = subprocess.run(
        [sys.executable, "-m", "gpulock", *args],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


service_dir = LOCK_ROOT / "service"
if service_dir.exists():
    shutil.rmtree(service_dir)

rc, out, err = run_cli(["service", "status"])
check("service status (no install) rc=4", rc == 4, f"rc={rc}")
check("service status (no install) reports installed: no",
      "installed:" in out and "no" in out, out.strip())

rc, out, err = run_cli(["service", "install", "--no-start", "--gpu-ids", "0,1",
                        "--idle-timeout", "600", "--no-placeholder-load",
                        "--env", "FOO=bar"])
check("service install --no-start rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check(
    "service install logs config path",
    "config saved to" in out,
    out.strip(),
)
check(
    "config.json written",
    (service_dir / "config.json").exists(),
    str(list(service_dir.iterdir())),
)
check(
    "supervisord.conf written",
    (service_dir / "supervisord.conf").exists(),
    str(list(service_dir.iterdir())),
)
saved_cfg = json.loads((service_dir / "config.json").read_text())
check("config.json gpu_ids", saved_cfg["gpu_ids"] == [0, 1], saved_cfg)
check("config.json idle_timeout", saved_cfg["idle_timeout"] == 600, saved_cfg)
check("config.json placeholder_load=False", saved_cfg["placeholder_load"] is False, saved_cfg)
check("config.json extra_env", saved_cfg["extra_env"].get("FOO") == "bar", saved_cfg)

conf_on_disk = (service_dir / "supervisord.conf").read_text()
for needle in (
    "[program:gpulock-guard]",
    "--idle-timeout 600",
    "--no-placeholder-load",
    'environment=FOO="bar"',
):
    check(
        f"supervisord.conf contains {needle!r}",
        needle in conf_on_disk,
        conf_on_disk[:200],
    )

rc, out, err = run_cli(["service", "status"])
check("service status (installed, stopped) rc=3", rc == 3, f"rc={rc}")
check("service status reports installed yes", "installed:    yes" in out, out.strip())
check("service status reports supervisord stopped",
      "supervisord:  stopped" in out, out.strip())

print("\n=== gpulock service config show / get / set / unset ===")
rc, out, err = run_cli(["service", "config", "path"])
check("config path rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("config path printed", out.strip().endswith("config.json"), out.strip())

rc, out, err = run_cli(["service", "config", "show"])
check("config show rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
for needle in ("gpu_ids=0,1", "idle_timeout=600", "placeholder_load=false"):
    check(f"config show contains {needle!r}", needle in out, out.strip())

rc, out, err = run_cli(["service", "config", "get", "idle_timeout"])
check("config get idle_timeout rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("config get idle_timeout==600", out.strip() == "600", out.strip())

rc, out, err = run_cli(["service", "config", "get", "bogus"])
check("config get bogus rc!=0", rc != 0, f"rc={rc}")

rc, out, err = run_cli([
    "service", "config", "set",
    "gpu_ids=2,3,4", "idle_timeout=1234", "placeholder_load=true",
])
check("config set rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("config set hints restart", "service restart" in out, out.strip())
saved_cfg2 = json.loads((service_dir / "config.json").read_text())
check("config set persisted gpu_ids", saved_cfg2["gpu_ids"] == [2, 3, 4], saved_cfg2)
check("config set persisted idle_timeout", saved_cfg2["idle_timeout"] == 1234, saved_cfg2)
check("config set persisted placeholder_load", saved_cfg2["placeholder_load"] is True, saved_cfg2)

rc, out, err = run_cli(["service", "config", "set", "idle_timeout=not-a-number"])
check("config set bad value rc!=0", rc != 0, f"rc={rc}")

rc, out, err = run_cli(["service", "config", "unset", "idle_timeout"])
check("config unset rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
saved_cfg3 = json.loads((service_dir / "config.json").read_text())
check("config unset restores default idle_timeout=5400",
      saved_cfg3["idle_timeout"] == 5400, saved_cfg3)

rc, out, err = run_cli(["service", "uninstall"])
check("service uninstall rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("config.json removed after uninstall", not (service_dir / "config.json").exists())
check(
    "supervisord.conf removed after uninstall",
    not (service_dir / "supervisord.conf").exists(),
)

print("\n=== install.sh sanity (no args allowed) ===")
install_sh = REPO / "install.sh"
p = subprocess.run(["bash", "-n", str(install_sh)], capture_output=True, text=True)
check("install.sh syntax OK", p.returncode == 0, p.stderr.strip())
p = subprocess.run(
    ["bash", str(install_sh), "--gpu-ids", "0"],
    capture_output=True, text=True,
)
check("install.sh rejects extra args (rc=2)", p.returncode == 2, f"rc={p.returncode}")
check(
    "install.sh suggests `gpulock service`",
    "gpulock service" in p.stderr,
    p.stderr.strip(),
)


print("\n=== supervisord daemon lifecycle (real fork) ===")
ok, sup_err = supervisor.supervisor_available()
if not ok:
    print(f"  [SKIP] supervisor package not importable: {sup_err}")
    print(f"         install with: {sys.executable} -m pip install --user 'supervisor>=4.2'")
else:
    if service_dir.exists():
        shutil.rmtree(service_dir)

    rc, out, err = run_cli([
        "service", "install", "--no-start",
        "--gpu-ids", "0",
        "--idle-timeout", "10",
        "--no-placeholder-load",
    ])
    check("install for daemon test rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")

    rc, out, err = run_cli(["service", "start"])
    check("service start rc=0", rc == 0,
          f"rc={rc} stdout={out.strip()} stderr={err.strip()}")

    sup_pid_path = service_dir / "supervisord.pid"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not sup_pid_path.exists():
        time.sleep(0.05)
    check("supervisord.pid file exists", sup_pid_path.exists(), str(sup_pid_path))

    sup_pid = int(sup_pid_path.read_text().strip()) if sup_pid_path.exists() else 0
    check("supervisord pid > 0", sup_pid > 0, f"pid={sup_pid}")

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    if sup_pid > 0:
        check("supervisord process alive", alive(sup_pid), f"pid={sup_pid}")
        time.sleep(0.5)
        check("supervisord still alive after 0.5s (detached)", alive(sup_pid))

    # Wait for supervisord to attempt at least one guard spawn (the guard.log
    # appears once supervisord opens the file).
    guard_log = service_dir / "guard.log"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not guard_log.exists():
        time.sleep(0.1)
    check("guard.log was created (supervisord spawned guard)", guard_log.exists(),
          str(guard_log))

    rc, out, err = run_cli(["service", "status"])
    # status returns 0 if supervisord is up AND program is RUNNING; otherwise
    # supervisorctl may report a non-RUNNING state. Either way supervisord must
    # be reported as running.
    check(
        "service status reports supervisord running",
        "supervisord:  running" in out,
        out.strip(),
    )

    if guard_log.exists():
        rc, out, err = run_cli(["service", "logs", "-n", "20"])
        check("service logs rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")

    rc, out, err = run_cli(["service", "stop"], timeout=40)
    check("service stop rc=0", rc == 0,
          f"rc={rc} stdout={out.strip()} stderr={err.strip()}")

    deadline = time.monotonic() + 5.0
    gone = False
    while time.monotonic() < deadline:
        if sup_pid > 0 and not alive(sup_pid):
            gone = True
            break
        time.sleep(0.1)
    check("supervisord process gone after stop", gone, f"pid={sup_pid}")
    check("supervisord.pid removed after stop", not sup_pid_path.exists())

    rc, out, err = run_cli(["service", "stop"])
    check("service stop is idempotent", rc == 0, f"rc={rc} out={out.strip()}")
    rc, out, err = run_cli(["service", "uninstall"])
    check("service uninstall after stop rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
    check(
        "config.json removed after final uninstall",
        not (service_dir / "config.json").exists(),
    )
    check(
        "supervisord.conf removed after final uninstall",
        not (service_dir / "supervisord.conf").exists(),
    )

print("\n=== summary ===")
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

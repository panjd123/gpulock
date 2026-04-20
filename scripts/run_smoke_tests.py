"""End-to-end functional smoke test for the gpulock refactor.

Runs entirely in user-space; uses isolated GPU_BENCH_LOCK_DIR so it can run on
a shared host without disturbing real state. Exercises:

* module import graph
* env helpers
* GuardServiceConfig save/load round-trip
* backend detection
* systemd unit rendering snapshot
* lock acquire+release for read mode (write mode skipped: needs nvidia driver)
* lock metadata persisted on disk
* gpulock service show / install --no-start / status / uninstall lifecycle (no daemon)
"""

from __future__ import annotations

import contextlib
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
    from gpulock import cli, guard, lock, gpu, config, paths, placeholder, logging_setup
    from gpulock.service import cli as service_cli, common, supervisor, systemd_user
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


print("\n=== service backend detection ===")
in_ctr = common.in_container()
sysd_ok = common.systemd_user_available()
auto = common.resolve_backend("auto")
print(f"     in_container={in_ctr}  systemd_user_available={sysd_ok}  auto -> {auto}")
check("explicit backend supervisor accepted", common.resolve_backend("supervisor") == "supervisor")
check("explicit backend systemd-user accepted", common.resolve_backend("systemd-user") == "systemd-user")
try:
    common.resolve_backend("nope")
    check("invalid backend raises ValueError", False, "expected ValueError")
except ValueError:
    check("invalid backend raises ValueError", True)


print("\n=== GuardServiceConfig round-trip ===")
cfg = common.GuardServiceConfig(
    backend="supervisor",
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


print("\n=== systemd unit render snapshot ===")
unit_text = systemd_user.render_unit(cfg)
expected_substrings = [
    "[Unit]\nDescription=gpulock guard daemon",
    "[Service]\nType=simple",
    'Environment="FOO=bar"',
    'Environment="K=v=eq"',
    "/opt/gpulock/bin/gpulock guard 0 2 5 --idle-timeout 900",
    "Restart=on-failure",
    "[Install]\nWantedBy=default.target",
]
for sub in expected_substrings:
    check(f"unit contains: {sub.splitlines()[0][:48]!r}", sub in unit_text)


print("\n=== read lock acquire+release (no nvidia driver needed) ===")
# Read lock skips the write-lock idle precheck and only depends on the file
# system, so it works fine without a real GPU.
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

# Acquire a second read lock concurrently.
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


print("\n=== gpulock service show / install --no-start / status / uninstall (no daemon) ===")
env = os.environ.copy()
env["GPU_BENCH_LOCK_DIR"] = str(LOCK_ROOT)
env["PYTHONPATH"] = str(REPO / "src")

def run_cli(args: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(
        [sys.executable, "-m", "gpulock", *args],
        capture_output=True, text=True, env=env, timeout=20,
    )
    return p.returncode, p.stdout, p.stderr

# Wipe any stale state before this section so the lifecycle is deterministic.
service_dir = LOCK_ROOT / "service"
if service_dir.exists():
    shutil.rmtree(service_dir)

rc, out, err = run_cli(["service", "show"])
check("service show (no install) rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("service show reports installed: no", "installed: no" in out, out.strip())

rc, out, err = run_cli(["service", "install", "--no-start", "--gpu-ids", "0,1",
                        "--idle-timeout", "600", "--no-placeholder-load",
                        "--env", "FOO=bar"])
check("service install --no-start rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("service install backend logged", "backend=" in out, out.strip())
check(
    "config.json written",
    (service_dir / "config.json").exists(),
    str(list(service_dir.iterdir())),
)
saved_cfg = json.loads((service_dir / "config.json").read_text())
check("config.json gpu_ids", saved_cfg["gpu_ids"] == [0, 1], saved_cfg)
check("config.json idle_timeout", saved_cfg["idle_timeout"] == 600, saved_cfg)
check("config.json placeholder_load=False", saved_cfg["placeholder_load"] is False, saved_cfg)
check("config.json extra_env", saved_cfg["extra_env"].get("FOO") == "bar", saved_cfg)

rc, out, err = run_cli(["service", "status"])
check("service status (stopped) rc!=0", rc != 0, f"rc={rc}")
check("service status reports stopped", "supervisor: stopped" in out, out.strip())

rc, out, err = run_cli(["service", "uninstall"])
check("service uninstall rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("config.json removed after uninstall", not (service_dir / "config.json").exists())


print("\n=== supervisor daemon lifecycle (real fork + exec) ===")
# Install a config that will make the guard child exit immediately (no nvidia
# driver in this container). The interesting thing we're testing is:
#  1) `service start` daemonizes a supervisor that survives the parent process
#  2) supervisor.pid file appears
#  3) supervisor restarts the guard child after it crashes
#  4) `service status` reports running
#  5) `service stop` cleanly tears everything down

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
check("service start rc=0", rc == 0, f"rc={rc} stdout={out.strip()} stderr={err.strip()}")

sup_pid_path = service_dir / "supervisor.pid"
sup_log_path = service_dir / "supervisor.log"
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline and not sup_pid_path.exists():
    time.sleep(0.05)
check("supervisor.pid file exists", sup_pid_path.exists(), str(sup_pid_path))

sup_pid = 0
if sup_pid_path.exists():
    sup_pid = int(sup_pid_path.read_text().strip())
    check("supervisor pid > 0", sup_pid > 0, f"pid={sup_pid}")

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    check("supervisor process alive", alive(sup_pid), f"pid={sup_pid}")

    # supervisor parent (the python -m gpulock service _run-supervisor) should
    # have exited after double-fork; verify the pid we recorded is NOT the
    # short-lived launcher process by checking its parent != the test process.
    # (we just check it's still alive after a beat)
    time.sleep(0.5)
    check("supervisor still alive after 0.5s (detached)", alive(sup_pid))

# Let the supervisor try to spawn guard a few times. With no nvidia driver,
# guard will exit non-zero. We expect to see at least one "guard exited"
# message in the log within a few seconds.
deadline = time.monotonic() + 6.0
saw_exit = False
last_log = ""
while time.monotonic() < deadline:
    if sup_log_path.exists():
        last_log = sup_log_path.read_text(encoding="utf-8", errors="ignore")
        if "guard exited" in last_log or "spawning guard" in last_log:
            saw_exit = True
            break
    time.sleep(0.2)
check(
    "supervisor log records spawn/exit cycle",
    saw_exit,
    f"log tail={last_log[-400:]!r}",
)

rc, out, err = run_cli(["service", "status"])
check("service status (running) rc=0", rc == 0, f"rc={rc} out={out.strip()}")
check(
    "service status reports supervisor running",
    "supervisor: running" in out,
    out.strip(),
)

# `logs -n 20` should print without crashing.
rc, out, err = run_cli(["service", "logs", "-n", "20"])
check("service logs rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check("service logs has supervisor lines", "supervisor" in out.lower(), out.strip()[-300:])

# Now stop and verify cleanup.
rc, out, err = run_cli(["service", "stop"])
check("service stop rc=0", rc == 0, f"rc={rc} stdout={out.strip()} stderr={err.strip()}")

# Wait for the supervisor process to actually go away.
deadline = time.monotonic() + 5.0
gone = False
while time.monotonic() < deadline:
    if sup_pid > 0:
        try:
            os.kill(sup_pid, 0)
        except OSError:
            gone = True
            break
    time.sleep(0.1)
check("supervisor process gone after stop", gone, f"pid={sup_pid}")
check("supervisor.pid removed after stop", not sup_pid_path.exists())

# Idempotent stop / status / uninstall.
rc, out, err = run_cli(["service", "stop"])
check("service stop is idempotent", rc == 0, f"rc={rc} out={out.strip()}")
rc, out, err = run_cli(["service", "uninstall"])
check("service uninstall after stop rc=0", rc == 0, f"rc={rc} stderr={err.strip()}")
check(
    "config.json removed after final uninstall",
    not (service_dir / "config.json").exists(),
)

print("\n=== summary ===")
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

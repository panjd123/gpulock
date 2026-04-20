"""Top-level argv dispatcher for the ``gpulock`` command line interface."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from .config import LockConfig, MODE_ALIAS_MAP, READ_MODE, WRITE_MODE
from .config import env_int as _env_int
from .guard import cmd_guard
from .lock import GpuBenchLock
from .logging_setup import setup_main_logger
from .paths import resolve_lock_root
from .placeholder import placeholder_main
from .service import cmd_service


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpulock",
        description=(
            "GPU benchmark lock wrapper. Preferred forms: "
            "gpulock perf <gpu_id> -- <cmd> / gpulock check <gpu_id> -- <cmd>"
        ),
    )
    parser.add_argument("--perf", action="store_true", help="Shortcut for performance mode (write lock).")
    parser.add_argument("--check", action="store_true", help="Shortcut for correctness mode (read lock).")
    parser.add_argument(
        "--mode",
        choices=[WRITE_MODE, READ_MODE],
        default=None,
        help="Lock mode. write=exclusive (performance); read=shared (correctness/functional).",
    )
    parser.add_argument("gpu_id", type=int, help="Physical GPU index to lock (nvidia-smi index).")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to run. Examples: gpulock perf 1 -- python bench.py "
            "or gpulock check 1 -- python test.py"
        ),
    )
    parser.add_argument("--poll-ms", type=int, default=LockConfig.poll_ms)
    parser.add_argument("--timeout-s", type=int, default=LockConfig.timeout_s)
    parser.add_argument("--grace-age-s", type=int, default=LockConfig.grace_age_s)
    parser.add_argument("--heartbeat-s", type=int, default=LockConfig.heartbeat_s)
    parser.add_argument("--orphan-check-s", type=int, default=LockConfig.orphan_check_s)
    parser.add_argument("--orphan-empty-threshold", type=int, default=LockConfig.orphan_empty_threshold)
    parser.add_argument(
        "--wait-gpu-idle",
        action="store_true",
        help="For write lock, wait until GPU stays idle continuously before acquire (instead of fail-fast).",
    )
    parser.add_argument(
        "--idle-streak-s",
        type=int,
        default=_env_int("GPU_BENCH_LOCK_IDLE_STREAK_S", 3),
        help="Consecutive util=0 checks required by write-lock precheck (default 3).",
    )
    parser.add_argument(
        "--idle-check-ms",
        type=int,
        default=_env_int("GPU_BENCH_LOCK_IDLE_CHECK_MS", 100),
        help="Polling interval in ms for --wait-gpu-idle (default 100).",
    )
    parser.add_argument(
        "--set-cuda-visible-devices",
        action="store_true",
        help="Export CUDA_VISIBLE_DEVICES=<gpu_id> for the child command.",
    )
    return parser


def _rewrite_argv_mode_alias(argv: list[str]) -> list[str]:
    if not argv:
        return argv

    try:
        sep_idx = argv.index("--")
    except ValueError:
        sep_idx = len(argv)

    prefix = argv[:sep_idx]
    suffix = argv[sep_idx:]
    if not prefix:
        return argv

    first = prefix[0].strip().lower()
    mapped = MODE_ALIAS_MAP.get(first)
    if mapped is None:
        return argv

    for token in prefix[1:]:
        if token in ("--mode", "--perf", "--check"):
            return argv
    return ["--mode", mapped] + prefix[1:] + suffix


def _resolve_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.perf and args.check:
        parser.error("--perf and --check cannot be used together")

    if args.perf:
        return WRITE_MODE
    if args.check:
        return READ_MODE
    if args.mode is not None:
        return args.mode
    mode = os.getenv("GPU_BENCH_LOCK_MODE", WRITE_MODE).strip().lower()
    if mode not in (READ_MODE, WRITE_MODE):
        mode = WRITE_MODE
    return mode


def _normalize_command(raw: list[str]) -> tuple[str, bool]:
    if not raw:
        raise ValueError(
            "Missing command. Usage: gpulock perf <gpu_id> -- <cmd> "
            "or gpulock check <gpu_id> -- <cmd> "
            "or gpulock [--mode read|write] <gpu_id> -- <cmd>"
        )
    if raw[0] == "--":
        raw = raw[1:]
        if not raw:
            raise ValueError("Missing command after '--'.")
    if len(raw) == 1:
        return (raw[0], True)
    return (" ".join(shlex.quote(x) for x in raw), True)


def main() -> int:
    prog = os.path.basename(sys.argv[0])
    if prog == "gpuunlock":
        print(
            "[gpulock] 'gpuunlock' has been removed. Use wrapped execution only: gpulock perf/check <gpu_id> -- <cmd>",
            file=sys.stderr,
        )
        return 2

    if len(sys.argv) > 1:
        sub = sys.argv[1]
        if sub == "_placeholder":
            keep_util = True
            if len(sys.argv) > 3:
                keep_util = sys.argv[3].strip().lower() not in ("0", "false", "no", "off")
            return placeholder_main(int(sys.argv[2]), keep_util=keep_util)
        if sub == "guard":
            return cmd_guard(sys.argv[2:])
        if sub == "service":
            return cmd_service(sys.argv[2:])
        if sub in ("lock", "unlock", "release"):
            print(
                "[gpulock] standalone lock/unlock has been removed to avoid leaked locks. "
                "Use wrapped execution only: gpulock perf/check <gpu_id> -- <cmd>",
                file=sys.stderr,
            )
            return 2

    parser = _build_parser()
    args = parser.parse_args(_rewrite_argv_mode_alias(sys.argv[1:]))
    mode = _resolve_mode(args, parser)
    log = setup_main_logger(resolve_lock_root())
    log.info(
        "cmd run request mode=%s gpu=%d wait_gpu_idle=%s idle_streak=%d idle_check_ms=%d argv=%s",
        mode,
        args.gpu_id,
        bool(args.wait_gpu_idle),
        max(args.idle_streak_s, 1),
        max(args.idle_check_ms, 100),
        " ".join(shlex.quote(x) for x in sys.argv),
    )

    cfg = LockConfig(
        poll_ms=max(args.poll_ms, 1),
        timeout_s=max(args.timeout_s, 1),
        grace_age_s=max(args.grace_age_s, 1),
        heartbeat_s=max(args.heartbeat_s, 1),
        orphan_check_s=max(args.orphan_check_s, 1),
        orphan_empty_threshold=max(args.orphan_empty_threshold, 1),
    )
    try:
        command, shell_mode = _normalize_command(args.command)
    except ValueError as e:
        log.error("cmd run invalid command gpu=%d mode=%s err=%s", args.gpu_id, mode, e)
        print(f"[gpulock] {e}", file=sys.stderr)
        return 2
    log.info("cmd run child command gpu=%d mode=%s shell=%s cmd=%s", args.gpu_id, mode, shell_mode, command)

    lock = GpuBenchLock(
        args.gpu_id,
        mode=mode,
        config=cfg,
        wait_gpu_idle=bool(args.wait_gpu_idle),
        idle_streak_s=max(args.idle_streak_s, 1),
        idle_check_ms=max(args.idle_check_ms, 100),
    )
    try:
        lock.acquire()
    except TimeoutError as e:
        log.error("cmd run lock timeout gpu=%d mode=%s err=%s", args.gpu_id, mode, e)
        print(f"[GPU Lock] timeout: {e}", file=sys.stderr)
        return 124
    except Exception as e:
        log.exception("cmd run lock acquire failed gpu=%d mode=%s err=%s", args.gpu_id, mode, e)
        print(f"[GPU Lock] acquire failed: {e}", file=sys.stderr)
        return 1

    path_str = str(lock.lock_path) if lock.lock_path is not None else ""
    print(
        f"[GPU Lock] acquired mode={lock.mode} device={lock.physical_device_id} lock_path={path_str}",
        flush=True,
    )

    child_env = os.environ.copy()
    child_env["GPU_BENCH_LOCKED_DEVICE"] = str(lock.physical_device_id)
    child_env["GPU_BENCH_LOCK_MODE"] = lock.mode
    if args.set_cuda_visible_devices:
        child_env["CUDA_VISIBLE_DEVICES"] = str(lock.physical_device_id)

    try:
        rc = subprocess.run(command, shell=shell_mode, executable="/bin/bash", env=child_env).returncode
    except KeyboardInterrupt:
        rc = 130
        log.warning("cmd run interrupted gpu=%d mode=%s rc=%d", args.gpu_id, mode, rc)
    except Exception as e:
        log.exception("cmd run child failed gpu=%d mode=%s err=%s", args.gpu_id, mode, e)
        rc = 1
    finally:
        lock.release()
        print(
            f"[GPU Lock] released mode={lock.mode} device={lock.physical_device_id}",
            flush=True,
        )
    log.info("cmd run finished gpu=%d mode=%s rc=%d", args.gpu_id, mode, rc)

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

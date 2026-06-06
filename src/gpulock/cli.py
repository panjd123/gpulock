"""Top-level argv dispatcher for the ``gpulock`` command line interface."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from .config import LockConfig, READ_MODE, WRITE_MODE
from .config import env_int as _env_int
from .guard import cmd_guard
from .logging_setup import setup_main_logger
from .paths import resolve_lock_root
from .placeholder import placeholder_main
from .service import cmd_service
from .session import MultiGpuLock


def _build_parser() -> argparse.ArgumentParser:
    env_config = LockConfig.from_env()
    parser = argparse.ArgumentParser(
        prog="gpulock",
        description=(
            "GPU read/write lock wrapper. Supported forms: "
            "gpulock perf <gpu_id> -- <cmd> / gpulock check <gpu_id> -- <cmd>; "
            "write and read are aliases for perf and check."
        ),
    )
    sub = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    def add_run_parser(name: str, mode: str, help_text: str) -> None:
        run_parser = sub.add_parser(name, help=help_text)
        run_parser.set_defaults(mode=mode)
        run_parser.add_argument(
            "gpu_ids",
            type=str,
            help="GPU indices: single int or comma-separated (e.g. 0,1,2).",
        )
        run_parser.add_argument("--poll-ms", type=int, default=env_config.poll_ms)
        run_parser.add_argument("--timeout-s", type=int, default=env_config.timeout_s)
        run_parser.add_argument("--grace-age-s", type=int, default=env_config.grace_age_s)
        run_parser.add_argument("--heartbeat-s", type=int, default=env_config.heartbeat_s)
        run_parser.add_argument(
            "--wait-gpu-idle",
            action="store_true",
            help="For perf, wait until GPU stays idle continuously before acquire.",
        )
        run_parser.add_argument(
            "--idle-streak-s",
            type=int,
            default=_env_int("GPULOCK_IDLE_STREAK_S", 3),
            help="Consecutive util=0 checks required by perf precheck (default 3).",
        )
        run_parser.add_argument(
            "--idle-check-ms",
            type=int,
            default=_env_int("GPULOCK_IDLE_CHECK_MS", 100),
            help="Polling interval in ms for --wait-gpu-idle (default 100).",
        )

    add_run_parser("perf", WRITE_MODE, "run a command with an exclusive write lock")
    add_run_parser("write", WRITE_MODE, "alias for perf")
    add_run_parser("check", READ_MODE, "run a command with a shared read lock")
    add_run_parser("read", READ_MODE, "alias for check")
    return parser


def _normalize_command(raw: list[str]) -> str:
    if not raw:
        raise ValueError(
            "Missing command. Usage: gpulock perf <gpu_ids> -- <cmd> "
            "or gpulock check <gpu_ids> -- <cmd>"
        )
    if raw[0] == "--":
        raw = raw[1:]
        if not raw:
            raise ValueError("Missing command after '--'.")
    if len(raw) == 1:
        return raw[0]
    return " ".join(shlex.quote(x) for x in raw)


def _parse_gpu_ids(raw: str) -> list[int]:
    """Parse '0,1,2' or '0' into sorted deduplicated list of GPU IDs."""
    parts = raw.replace(" ", "").split(",")
    ids = sorted(set(int(p) for p in parts if p))
    if not ids:
        raise ValueError("No valid GPU IDs provided")
    return ids


def _parse_run_args(argv: list[str]) -> argparse.Namespace:
    parser = _build_parser()
    try:
        sep_idx = argv.index("--")
    except ValueError:
        args = parser.parse_args(argv)
        args.command = []
        return args

    gpulock_argv = argv[:sep_idx]
    user_cmd = argv[sep_idx:]
    args = parser.parse_args(gpulock_argv)
    args.command = user_cmd
    return args


def _run_locked_command(args: argparse.Namespace) -> int:
    mode = args.mode
    log = setup_main_logger(resolve_lock_root())

    try:
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
    except ValueError as e:
        print(f"[gpulock] {e}", file=sys.stderr)
        return 2

    gpu_ids_str = ",".join(str(g) for g in gpu_ids)
    log.info(
        "cmd run request mode=%s gpus=%s wait_gpu_idle=%s idle_streak=%d idle_check_ms=%d argv=%s",
        mode,
        gpu_ids_str,
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
    )
    try:
        command = _normalize_command(args.command)
    except ValueError as e:
        log.error("cmd run invalid command gpus=%s mode=%s err=%s", gpu_ids_str, mode, e)
        print(f"[gpulock] {e}", file=sys.stderr)
        return 2
    log.info(
        "cmd run child command gpus=%s mode=%s cmd=%s",
        gpu_ids_str,
        mode,
        command,
    )

    session = MultiGpuLock(
        gpu_ids=gpu_ids,
        mode=mode,
        config=cfg,
        wait_gpu_idle=bool(args.wait_gpu_idle),
        idle_streak_s=max(args.idle_streak_s, 1),
        idle_check_ms=max(args.idle_check_ms, 100),
    )
    try:
        session.acquire()
    except TimeoutError as e:
        log.error("cmd run lock timeout gpus=%s mode=%s err=%s", gpu_ids_str, mode, e)
        print(f"[GPU Lock] timeout: {e}", file=sys.stderr)
        return 124
    except Exception as e:
        log.exception("cmd run lock acquire failed gpus=%s mode=%s err=%s", gpu_ids_str, mode, e)
        print(f"[GPU Lock] acquire failed: {e}", file=sys.stderr)
        return 1

    session.register_process_cleanup()
    print(
        f"[GPU Lock] acquired mode={mode} devices={gpu_ids_str} lock_paths={session.lock_paths_str()}",
        flush=True,
    )

    child_env = os.environ.copy()
    child_env.update(session.child_env_overrides())

    try:
        rc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            env=child_env,
        ).returncode
    except KeyboardInterrupt:
        rc = 130
        log.warning("cmd run interrupted gpus=%s mode=%s rc=%d", gpu_ids_str, mode, rc)
    except Exception as e:
        log.exception("cmd run child failed gpus=%s mode=%s err=%s", gpu_ids_str, mode, e)
        rc = 1
    finally:
        session.release()
        print(f"[GPU Lock] released mode={mode} devices={gpu_ids_str}", flush=True)
    log.info("cmd run finished gpus=%s mode=%s rc=%d", gpu_ids_str, mode, rc)

    return int(rc)


def main() -> int:
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

    args = _parse_run_args(sys.argv[1:])
    return _run_locked_command(args)


if __name__ == "__main__":
    raise SystemExit(main())

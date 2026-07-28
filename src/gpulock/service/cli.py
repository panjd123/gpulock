"""Argument parsing for ``gpulock service ...`` subcommands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Callable

from ..agent import install_global_agent_policies
from ..gpu import gpu_indices
from ..config import (
    DEFAULT_PLACEHOLDER_RELEASE_MODE,
    PLACEHOLDER_RELEASE_MODES,
    normalize_placeholder_release_mode,
)
from . import supervisor as supervisor_backend
from .common import (
    DEFAULT_GUARD_POLL_S,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PLACEHOLDER_IDLE_S,
    DEFAULT_PLACEHOLDER_MEM_RATIO,
    GuardServiceConfig,
    say,
    warn,
)


# --- value parsers --------------------------------------------------------

def _parse_gpu_ids(raw: str | None) -> list[int]:
    if raw is None:
        return []
    items: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            items.append(int(token))
        except ValueError:
            raise ValueError(f"invalid GPU id: {token!r}") from None
    return items


def _parse_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on", "y", "t"):
        return True
    if s in ("0", "false", "no", "off", "n", "f"):
        return False
    raise ValueError(f"expected bool, got {raw!r}")


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--env has empty key in {item!r}")
        env[k] = v
    return env


# Settable config keys (via `gpulock service config set/get/unset`),
# mapped to (parser, default-factory).
def _parse_mem_ratio(raw: str) -> float:
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"placeholder_mem_ratio must be between 0.0 and 1.0, got {value}")
    return value


_CONFIG_KEYS: dict[str, tuple[Callable[[str], Any], Callable[[], Any]]] = {
    "gpu_ids": (_parse_gpu_ids, list),
    "idle_timeout": (int, lambda: DEFAULT_IDLE_TIMEOUT),
    "placeholder_idle_s": (float, lambda: DEFAULT_PLACEHOLDER_IDLE_S),
    "guard_poll_s": (float, lambda: DEFAULT_GUARD_POLL_S),
    "placeholder_mem_ratio": (_parse_mem_ratio, lambda: DEFAULT_PLACEHOLDER_MEM_RATIO),
    "placeholder_release_mode": (normalize_placeholder_release_mode, lambda: DEFAULT_PLACEHOLDER_RELEASE_MODE),
}
_HANDY_IDLE_TIMEOUT = 315360000
_CONFIG_PRESETS = ("handy", "all", "default")


def _validate_key(key: str) -> None:
    """Print a helpful error and abort with rc=2 if ``key`` isn't settable."""
    if key not in _CONFIG_KEYS:
        warn(
            f"unknown key {key!r}; "
            f"settable keys: {', '.join(sorted(_CONFIG_KEYS))}"
        )
        raise SystemExit(2)


def _format_value(key: str, value: Any) -> str:
    if key == "gpu_ids":
        return ",".join(str(x) for x in value) if value else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_cfg() -> GuardServiceConfig:
    """Load the persisted config, or abort with rc=2 if it's not there."""
    try:
        return GuardServiceConfig.load()
    except FileNotFoundError as e:
        warn(str(e))
        raise SystemExit(2)


# --- argparse -------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    keys_help = ", ".join(sorted(_CONFIG_KEYS))
    parser = argparse.ArgumentParser(
        prog="gpulock service",
        description="Install and manage `gpulock guard` as a supervisord-managed service.",
    )
    sub = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    p_install = sub.add_parser(
        "install", help="write config + supervisord.conf and (by default) start the service",
    )
    p_install.add_argument(
        "--gpu-ids", default=None,
        help="comma/space separated GPU IDs to watch (default: all visible GPUs)",
    )
    p_install.add_argument("--idle-timeout", type=int, default=DEFAULT_IDLE_TIMEOUT)
    p_install.add_argument("--placeholder-idle-s", type=float, default=DEFAULT_PLACEHOLDER_IDLE_S)
    p_install.add_argument("--guard-poll-s", type=float, default=DEFAULT_GUARD_POLL_S)
    p_install.add_argument("--placeholder-mem-ratio", type=float, default=DEFAULT_PLACEHOLDER_MEM_RATIO,
                           help="fraction of GPU memory to allocate (0.0-1.0, 0 = compute-only)")
    p_install.add_argument(
        "--placeholder-release-mode",
        default=DEFAULT_PLACEHOLDER_RELEASE_MODE,
        choices=PLACEHOLDER_RELEASE_MODES,
        help=(
            "placeholder release behavior when real work is detected: "
            "stop = destroy CUDA context before work (clean, slower restart; default), "
            "park = legacy fast mode with resident CUDA context"
        ),
    )
    p_install.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="extra environment variable to inject into the guard (repeat to add more)",
    )
    p_install.add_argument(
        "--no-start", action="store_true",
        help="don't start the service immediately after installing",
    )
    p_install.add_argument(
        "--no-agent-policy", action="store_true",
        help="skip installing the gpulock policy block into global AGENTS.md files",
    )

    sub.add_parser("uninstall", help="stop the service and remove its config")
    sub.add_parser("start", help="start supervisord (and the guard program)")
    sub.add_parser("stop", help="stop supervisord (and the guard program)")
    sub.add_parser("restart", help="stop + regenerate conf + start")
    sub.add_parser("status", help="show service / supervisord / guard status")

    p_logs = sub.add_parser("logs", help="tail the guard logs")
    p_logs.add_argument("-n", "--lines", type=int, default=200)
    p_logs.add_argument("-f", "--follow", action="store_true")

    # `gpulock service config <action>` — manage guard runtime config
    p_config = sub.add_parser(
        "config",
        help="show / modify the guard service config (apply with `service restart`)",
    )
    cfg_sub = p_config.add_subparsers(dest="config_action", metavar="ACTION", required=True)
    cfg_sub.add_parser("show", help="print current config (key=value)")
    cfg_sub.add_parser("path", help="print the config file path")
    p_get = cfg_sub.add_parser("get", help="print one config value")
    p_get.add_argument("key", help=f"one of: {keys_help}")
    p_set = cfg_sub.add_parser("set", help="set one or more config values")
    p_set.add_argument("kv", nargs="+", metavar="KEY=VALUE", help=f"settable keys: {keys_help}")
    p_unset = cfg_sub.add_parser("unset", help="reset one config value to its default")
    p_unset.add_argument("key", help=f"one of: {keys_help}")
    p_preset = cfg_sub.add_parser("preset", help="apply a named config preset")
    p_preset.add_argument("name", choices=_CONFIG_PRESETS, help="preset name")
    cfg_sub.add_parser("edit", help="open the config file in $EDITOR")

    return parser


# --- action handlers ------------------------------------------------------

def _do_install(args: argparse.Namespace) -> int:
    try:
        cfg = GuardServiceConfig(
            gpu_ids=_parse_gpu_ids(args.gpu_ids),
            idle_timeout=int(args.idle_timeout),
            placeholder_idle_s=float(args.placeholder_idle_s),
            guard_poll_s=float(args.guard_poll_s),
            placeholder_mem_ratio=_parse_mem_ratio(str(args.placeholder_mem_ratio)),
            placeholder_release_mode=normalize_placeholder_release_mode(args.placeholder_release_mode),
            extra_env=_parse_env_kv(list(args.env)),
            python_executable=sys.executable or "",
            gpulock_executable=shutil.which("gpulock") or "",
        )
    except ValueError as e:
        warn(str(e))
        return 2
    rc = supervisor_backend.install(cfg, start_now=not args.no_start)
    say(f"config saved to {GuardServiceConfig.config_path()}")
    say(f"supervisord conf at {supervisor_backend.conf_path()}")
    if not args.no_agent_policy:
        for path, changed in install_global_agent_policies():
            status = "updated" if changed else "unchanged"
            say(f"agent policy {status}: {path}")
    if args.no_start:
        say("not started (--no-start). start later with `gpulock service start`.")
    return rc


def _config_show(_args: argparse.Namespace) -> int:
    cfg = _load_cfg()
    print(f"# {GuardServiceConfig.config_path()}")
    for key in sorted(_CONFIG_KEYS):
        print(f"{key}={_format_value(key, getattr(cfg, key))}")
    if cfg.extra_env:
        print(f"# extra_env (use `config edit` to modify): {cfg.extra_env}")
    return 0


def _config_path(_args: argparse.Namespace) -> int:
    print(GuardServiceConfig.config_path())
    return 0


def _config_get(args: argparse.Namespace) -> int:
    _validate_key(args.key)
    cfg = _load_cfg()
    print(_format_value(args.key, getattr(cfg, args.key)))
    return 0


def _config_set(args: argparse.Namespace) -> int:
    cfg = _load_cfg()
    for item in args.kv:
        if "=" not in item:
            warn(f"expected KEY=VALUE, got {item!r}")
            return 2
        key, raw = item.split("=", 1)
        key = key.strip()
        _validate_key(key)
        parser_fn, _ = _CONFIG_KEYS[key]
        try:
            value = parser_fn(raw)
        except ValueError as e:
            warn(f"invalid value for {key}: {e}")
            return 2
        setattr(cfg, key, value)
    cfg.save()
    say(f"config updated: {GuardServiceConfig.config_path()}")
    say("apply with: gpulock service restart")
    return 0


def _config_unset(args: argparse.Namespace) -> int:
    _validate_key(args.key)
    cfg = _load_cfg()
    _, default_factory = _CONFIG_KEYS[args.key]
    setattr(cfg, args.key, default_factory())
    cfg.save()
    say(f"reset {args.key} to default")
    say("apply with: gpulock service restart")
    return 0


def _handy_gpu_ids() -> list[int]:
    ids = sorted(set(gpu_indices()))
    if not ids:
        raise ValueError("no GPUs found via nvidia-smi")
    if len(ids) == 1:
        return ids
    return ids[1:]


def _all_gpu_ids() -> list[int]:
    ids = sorted(set(gpu_indices()))
    if not ids:
        raise ValueError("no GPUs found via nvidia-smi")
    return ids


def _config_preset(args: argparse.Namespace) -> int:
    cfg = _load_cfg()
    if args.name == "handy":
        try:
            cfg.gpu_ids = _handy_gpu_ids()
        except ValueError as e:
            warn(f"cannot apply preset {args.name!r}: {e}")
            return 2
        cfg.idle_timeout = _HANDY_IDLE_TIMEOUT
        cfg.placeholder_idle_s = DEFAULT_PLACEHOLDER_IDLE_S
        cfg.guard_poll_s = DEFAULT_GUARD_POLL_S
        cfg.placeholder_mem_ratio = 0.0
        cfg.placeholder_release_mode = DEFAULT_PLACEHOLDER_RELEASE_MODE
    elif args.name == "all":
        try:
            cfg.gpu_ids = _all_gpu_ids()
        except ValueError as e:
            warn(f"cannot apply preset {args.name!r}: {e}")
            return 2
        cfg.idle_timeout = _HANDY_IDLE_TIMEOUT
        cfg.placeholder_idle_s = DEFAULT_PLACEHOLDER_IDLE_S
        cfg.guard_poll_s = DEFAULT_GUARD_POLL_S
        cfg.placeholder_mem_ratio = 0.0
        cfg.placeholder_release_mode = DEFAULT_PLACEHOLDER_RELEASE_MODE
    elif args.name == "default":
        cfg.gpu_ids = []
        cfg.idle_timeout = DEFAULT_IDLE_TIMEOUT
        cfg.placeholder_idle_s = DEFAULT_PLACEHOLDER_IDLE_S
        cfg.guard_poll_s = DEFAULT_GUARD_POLL_S
        cfg.placeholder_mem_ratio = DEFAULT_PLACEHOLDER_MEM_RATIO
        cfg.placeholder_release_mode = DEFAULT_PLACEHOLDER_RELEASE_MODE
    else:
        warn(f"unknown preset {args.name!r}; presets: {', '.join(_CONFIG_PRESETS)}")
        return 2
    cfg.save()
    say(
        f"applied preset {args.name}: "
        f"gpu_ids={_format_value('gpu_ids', cfg.gpu_ids)} "
        f"idle_timeout={cfg.idle_timeout} "
        f"placeholder_mem_ratio={cfg.placeholder_mem_ratio} "
        f"placeholder_release_mode={cfg.placeholder_release_mode}"
    )
    say("apply with: gpulock service restart")
    return 0


def _config_edit(_args: argparse.Namespace) -> int:
    path = GuardServiceConfig.config_path()
    if not path.exists():
        warn(f"no config at {path}. run `gpulock service install --no-start` first.")
        return 2
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    try:
        ret = subprocess.call([editor, str(path)])
    except FileNotFoundError:
        warn(f"editor not found: {editor!r}")
        return 2
    if ret == 0:
        try:
            GuardServiceConfig.load()  # validate
        except Exception as e:  # noqa: BLE001
            warn(f"WARNING: config did not parse cleanly: {e}")
            return 1
        say("config saved. apply with: gpulock service restart")
    return ret


_CONFIG_ACTIONS: dict[str, Callable[[argparse.Namespace], int]] = {
    "show": _config_show,
    "path": _config_path,
    "get": _config_get,
    "set": _config_set,
    "unset": _config_unset,
    "preset": _config_preset,
    "edit": _config_edit,
}


def _do_config(args: argparse.Namespace) -> int:
    return _CONFIG_ACTIONS[args.config_action](args)


# --- top-level dispatch ---------------------------------------------------

_ACTIONS: dict[str, Callable[[argparse.Namespace], int]] = {
    "install":   _do_install,
    "uninstall": lambda _a: supervisor_backend.uninstall(),
    "start":     lambda _a: supervisor_backend.start(),
    "stop":      lambda _a: supervisor_backend.stop(),
    "restart":   lambda _a: supervisor_backend.restart(),
    "status":    lambda _a: supervisor_backend.status(),
    "logs":      lambda a: supervisor_backend.logs(lines=a.lines, follow=a.follow),
    "config":    _do_config,
}


def cmd_service(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    return _ACTIONS[args.action](args)

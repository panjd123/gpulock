"""Argument parsing for ``gpulock service ...`` subcommands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

from . import supervisor as supervisor_backend
from . import systemd_user as systemd_backend
from .common import (
    AUTO_BACKEND,
    SUPERVISOR_BACKEND,
    SUPPORTED_BACKENDS,
    SYSTEMD_USER_BACKEND,
    GuardServiceConfig,
    in_container,
    resolve_backend,
    service_dir,
    systemd_user_available,
)


def _parse_gpu_ids(raw: Optional[str]) -> list[int]:
    if raw is None:
        return []
    items: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            items.append(int(token))
        except ValueError:
            raise SystemExit(f"[gpulock service] invalid GPU id: {token!r}")
    return items


def _parse_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on", "y", "t"):
        return True
    if s in ("0", "false", "no", "off", "n", "f"):
        return False
    raise ValueError(f"expected bool, got {raw!r}")


def _parse_backend_value(raw: str) -> str:
    s = raw.strip()
    if s in (AUTO_BACKEND, *SUPPORTED_BACKENDS):
        return s
    raise ValueError(
        f"expected one of: {AUTO_BACKEND}/{'/'.join(SUPPORTED_BACKENDS)}, got {raw!r}"
    )


# Settable config keys (via `gpulock service config set/get/unset`),
# mapped to their parser and the default to restore on `unset`.
_CONFIG_KEYS: dict[str, tuple[Callable[[str], Any], Any]] = {
    "backend": (_parse_backend_value, SUPERVISOR_BACKEND),
    "gpu_ids": (_parse_gpu_ids, []),
    "idle_timeout": (int, 5400),
    "placeholder_idle_s": (float, 0.0),
    "placeholder_load": (_parse_bool, True),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpulock service",
        description="Install and manage `gpulock guard` as a service.",
    )
    sub = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    p_install = sub.add_parser("install", help="install gpulock guard as a service and start it")
    p_install.add_argument(
        "--backend",
        choices=[AUTO_BACKEND, *SUPPORTED_BACKENDS],
        default=AUTO_BACKEND,
        help="service backend (default: auto-detect)",
    )
    p_install.add_argument(
        "--gpu-ids", default=None,
        help="comma/space separated GPU IDs to watch (default: all visible GPUs)",
    )
    p_install.add_argument("--idle-timeout", type=int, default=5400)
    p_install.add_argument("--placeholder-idle-s", type=float, default=0.0)
    g = p_install.add_mutually_exclusive_group()
    g.add_argument("--placeholder-load", dest="placeholder_load", action="store_true", default=True)
    g.add_argument("--no-placeholder-load", dest="placeholder_load", action="store_false")
    p_install.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable to inject into the service (repeat to add more)",
    )
    p_install.add_argument(
        "--no-start", action="store_true",
        help="don't start the service immediately after installing",
    )
    p_install.add_argument(
        "--no-enable", action="store_true",
        help="(systemd-user backend) don't enable autostart at login",
    )

    sub.add_parser("uninstall", help="stop and remove the gpulock guard service")
    sub.add_parser("start", help="start the gpulock guard service")
    sub.add_parser("stop", help="stop the gpulock guard service")
    sub.add_parser("restart", help="restart the gpulock guard service")
    sub.add_parser("status", help="show service status")
    sub.add_parser("enable", help="(systemd-user backend) enable autostart")
    sub.add_parser("disable", help="(systemd-user backend) disable autostart")

    p_logs = sub.add_parser("logs", help="tail the service logs")
    p_logs.add_argument("-n", "--lines", type=int, default=200)
    p_logs.add_argument("-f", "--follow", action="store_true")

    p_show = sub.add_parser("show", help="show resolved service configuration / detection")

    # `gpulock service config <action>` — manage guard runtime config
    p_config = sub.add_parser(
        "config",
        help="show / modify the guard service config (apply with `service restart`)",
    )
    cfg_sub = p_config.add_subparsers(dest="config_action", metavar="ACTION", required=True)
    cfg_sub.add_parser("show", help="print current config (key=value)")
    cfg_sub.add_parser("path", help="print the config file path")
    p_get = cfg_sub.add_parser("get", help="print one config value")
    p_get.add_argument("key", help=f"one of: {', '.join(sorted(_CONFIG_KEYS))}")
    p_set = cfg_sub.add_parser("set", help="set one or more config values")
    p_set.add_argument(
        "kv", nargs="+", metavar="KEY=VALUE",
        help=f"settable keys: {', '.join(sorted(_CONFIG_KEYS))}",
    )
    p_unset = cfg_sub.add_parser("unset", help="reset one config value to its default")
    p_unset.add_argument("key", help=f"one of: {', '.join(sorted(_CONFIG_KEYS))}")
    cfg_sub.add_parser("edit", help="open the config file in $EDITOR")

    # Internal: long-running supervisor process, started by `service start`.
    p_run = sub.add_parser("_run-supervisor", help=argparse.SUPPRESS)
    return parser


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"[gpulock service] --env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"[gpulock service] --env has empty key in {item!r}")
        env[k] = v
    return env


def _resolve_gpulock_executable() -> str:
    """Best-effort: a stable path to the installed `gpulock` binary, if any."""
    found = shutil.which("gpulock")
    return found or ""


def _do_install(args: argparse.Namespace) -> int:
    backend = resolve_backend(args.backend)
    cfg = GuardServiceConfig(
        backend=backend,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        idle_timeout=int(args.idle_timeout),
        placeholder_idle_s=float(args.placeholder_idle_s),
        placeholder_load=bool(args.placeholder_load),
        extra_env=_parse_env_kv(list(args.env)),
        python_executable=sys.executable or "",
        gpulock_executable=_resolve_gpulock_executable(),
    )
    cfg.save()

    print(f"[gpulock service] backend={backend}")
    print(f"[gpulock service] config saved to {GuardServiceConfig.config_path()}")

    if backend == SYSTEMD_USER_BACKEND:
        unit = systemd_backend.install(cfg, start=not args.no_start, enable=not args.no_enable)
        print(f"[gpulock service] systemd unit installed at {unit}")
        if not args.no_start:
            print(f"[gpulock service] use `systemctl --user status {systemd_backend.UNIT_FILENAME}` to inspect")
        if args.no_enable:
            print("[gpulock service] autostart NOT enabled (--no-enable). enable later with `gpulock service enable`.")
        else:
            print(
                "[gpulock service] note: for systemd --user services to run when you're not "
                "logged in, enable lingering: `loginctl enable-linger $(whoami)`."
            )
        return 0

    if backend == SUPERVISOR_BACKEND:
        if not args.no_start:
            return supervisor_backend.start()
        print("[gpulock service] not started (--no-start). start later with `gpulock service start`.")
        return 0

    print(f"[gpulock service] unsupported backend: {backend}", file=sys.stderr)
    return 2


def _load_backend_or_exit() -> tuple[str, GuardServiceConfig]:
    cfg = GuardServiceConfig.load()
    return cfg.backend, cfg


def _do_uninstall(_args: argparse.Namespace) -> int:
    try:
        backend, _cfg = _load_backend_or_exit()
    except FileNotFoundError as e:
        print(f"[gpulock service] {e}", file=sys.stderr)
        return 0
    if backend == SYSTEMD_USER_BACKEND:
        systemd_backend.uninstall()
    elif backend == SUPERVISOR_BACKEND:
        supervisor_backend.uninstall()
    else:
        print(f"[gpulock service] unknown backend in config: {backend}", file=sys.stderr)
        return 2
    GuardServiceConfig.config_path().unlink(missing_ok=True)
    print("[gpulock service] uninstalled")
    return 0


def _backend_dispatch(action: str) -> int:
    backend, _cfg = _load_backend_or_exit()
    if backend == SYSTEMD_USER_BACKEND:
        m = {
            "start": systemd_backend.start,
            "stop": systemd_backend.stop,
            "restart": systemd_backend.restart,
            "status": systemd_backend.status,
            "enable": systemd_backend.enable,
            "disable": systemd_backend.disable,
        }
    elif backend == SUPERVISOR_BACKEND:
        m = {
            "start": supervisor_backend.start,
            "stop": supervisor_backend.stop,
            "restart": supervisor_backend.restart,
            "status": supervisor_backend.status,
            "enable": lambda: (print("[gpulock service] enable is a no-op for supervisor backend (already auto-restart-on-crash)"), 0)[1],
            "disable": lambda: (print("[gpulock service] disable is a no-op for supervisor backend; use `gpulock service stop`"), 0)[1],
        }
    else:
        print(f"[gpulock service] unknown backend: {backend}", file=sys.stderr)
        return 2
    fn = m.get(action)
    if fn is None:
        print(f"[gpulock service] action {action!r} not supported by backend {backend!r}", file=sys.stderr)
        return 2
    return int(fn() or 0)


def _do_logs(args: argparse.Namespace) -> int:
    backend, _cfg = _load_backend_or_exit()
    lines = max(int(args.lines), 1)
    if backend == SYSTEMD_USER_BACKEND:
        return systemd_backend.logs(lines=lines, follow=args.follow)
    if backend == SUPERVISOR_BACKEND:
        return supervisor_backend.logs(lines=lines, follow=args.follow)
    return 2


def _format_config_value(key: str, value: Any) -> str:
    if key == "gpu_ids":
        return ",".join(str(x) for x in value) if value else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_cfg_for_config_cmd() -> GuardServiceConfig:
    try:
        return GuardServiceConfig.load()
    except FileNotFoundError as e:
        raise SystemExit(f"[gpulock service] {e}")


def _do_config(args: argparse.Namespace) -> int:
    sub = args.config_action

    if sub == "path":
        print(GuardServiceConfig.config_path())
        return 0

    if sub == "show":
        cfg = _load_cfg_for_config_cmd()
        print(f"# {GuardServiceConfig.config_path()}")
        for key in sorted(_CONFIG_KEYS):
            print(f"{key}={_format_config_value(key, getattr(cfg, key))}")
        if cfg.extra_env:
            print(f"# extra_env (use `config edit` to modify): {cfg.extra_env}")
        return 0

    if sub == "get":
        cfg = _load_cfg_for_config_cmd()
        if args.key not in _CONFIG_KEYS:
            print(
                f"[gpulock service] unknown key {args.key!r}; "
                f"settable keys: {', '.join(sorted(_CONFIG_KEYS))}",
                file=sys.stderr,
            )
            return 2
        print(_format_config_value(args.key, getattr(cfg, args.key)))
        return 0

    if sub == "set":
        cfg = _load_cfg_for_config_cmd()
        backend_changed_from = None
        for item in args.kv:
            if "=" not in item:
                print(f"[gpulock service] expected KEY=VALUE, got {item!r}", file=sys.stderr)
                return 2
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in _CONFIG_KEYS:
                print(
                    f"[gpulock service] unknown key {key!r}; "
                    f"settable keys: {', '.join(sorted(_CONFIG_KEYS))}",
                    file=sys.stderr,
                )
                return 2
            parser_fn, _default = _CONFIG_KEYS[key]
            try:
                value = parser_fn(raw)
            except (ValueError, SystemExit) as e:
                print(f"[gpulock service] invalid value for {key}: {e}", file=sys.stderr)
                return 2
            if key == "backend" and value != cfg.backend:
                backend_changed_from = cfg.backend
            setattr(cfg, key, value)
        cfg.save()
        print(f"[gpulock service] config updated: {GuardServiceConfig.config_path()}")
        if backend_changed_from is not None:
            print(
                f"[gpulock service] WARNING: backend changed ({backend_changed_from} -> {cfg.backend}); "
                "you must run `gpulock service uninstall` (with the old backend active) "
                "and then `gpulock service install --no-start` to switch.",
                file=sys.stderr,
            )
        else:
            print("[gpulock service] apply with: gpulock service restart")
        return 0

    if sub == "unset":
        cfg = _load_cfg_for_config_cmd()
        if args.key not in _CONFIG_KEYS:
            print(
                f"[gpulock service] unknown key {args.key!r}; "
                f"settable keys: {', '.join(sorted(_CONFIG_KEYS))}",
                file=sys.stderr,
            )
            return 2
        _parser, default = _CONFIG_KEYS[args.key]
        setattr(cfg, args.key, default() if callable(default) else default)
        cfg.save()
        print(f"[gpulock service] reset {args.key} to default")
        print("[gpulock service] apply with: gpulock service restart")
        return 0

    if sub == "edit":
        path = GuardServiceConfig.config_path()
        if not path.exists():
            print(
                f"[gpulock service] no config at {path}. run `gpulock service install --no-start` first.",
                file=sys.stderr,
            )
            return 2
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
        try:
            ret = subprocess.call([editor, str(path)])
        except FileNotFoundError:
            print(f"[gpulock service] editor not found: {editor!r}", file=sys.stderr)
            return 2
        if ret == 0:
            try:
                GuardServiceConfig.load()  # validate
            except Exception as e:  # noqa: BLE001
                print(f"[gpulock service] WARNING: config did not parse cleanly: {e}", file=sys.stderr)
                return 1
            print("[gpulock service] config saved. apply with: gpulock service restart")
        return ret

    print(f"[gpulock service] unknown config action: {sub}", file=sys.stderr)
    return 2


def _do_show(_args: argparse.Namespace) -> int:
    in_ctr = in_container()
    sysd_ok = systemd_user_available()
    print(f"detection: in_container={in_ctr} systemd_user_available={sysd_ok}")
    print(f"resolved auto-backend: {resolve_backend(AUTO_BACKEND)}")
    cfg_path = GuardServiceConfig.config_path()
    if not cfg_path.exists():
        print(f"installed: no (missing config at {cfg_path})")
        return 0
    cfg = GuardServiceConfig.load()
    print(f"installed: yes (config={cfg_path})")
    print(f"  backend: {cfg.backend}")
    print(f"  gpu_ids: {cfg.gpu_ids or '<all visible GPUs>'}")
    print(f"  idle_timeout: {cfg.idle_timeout}s")
    print(f"  placeholder_idle_s: {cfg.placeholder_idle_s}")
    print(f"  placeholder_load: {cfg.placeholder_load}")
    if cfg.extra_env:
        print(f"  extra_env: {cfg.extra_env}")
    print(f"  python_executable: {cfg.python_executable}")
    print(f"  gpulock_executable: {cfg.gpulock_executable or '<use python -m gpulock>'}")
    print(f"  service_dir: {service_dir()}")
    return 0


def cmd_service(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    action = args.action
    if action == "install":
        return _do_install(args)
    if action == "uninstall":
        return _do_uninstall(args)
    if action in ("start", "stop", "restart", "status", "enable", "disable"):
        return _backend_dispatch(action)
    if action == "logs":
        return _do_logs(args)
    if action == "show":
        return _do_show(args)
    if action == "config":
        return _do_config(args)
    if action == "_run-supervisor":
        return supervisor_backend.run_supervisor()
    parser.error(f"unknown action: {action}")
    return 2  # unreachable

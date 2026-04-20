"""Backend detection and shared service helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..paths import resolve_lock_root


SUPERVISOR_BACKEND = "supervisor"
SYSTEMD_USER_BACKEND = "systemd-user"
SUPPORTED_BACKENDS = (SUPERVISOR_BACKEND, SYSTEMD_USER_BACKEND)
AUTO_BACKEND = "auto"

SERVICE_NAME = "gpulock-guard"


def service_dir(lock_root: Optional[Path] = None) -> Path:
    """Directory holding service config / pid / log files."""
    root = lock_root or resolve_lock_root()
    path = root / "service"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def in_container() -> bool:
    """Best-effort detection of "we're inside a container"."""
    if Path("/.dockerenv").exists():
        return True
    if Path("/run/.containerenv").exists():
        return True
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return True
    cgroup = Path("/proc/1/cgroup")
    try:
        text = cgroup.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    needles = ("docker", "containerd", "kubepods", "crio", "podman", "lxc")
    lower = text.lower()
    return any(needle in lower for needle in needles)


def systemd_user_available() -> bool:
    """Return True if ``systemctl --user`` looks usable for this user."""
    if shutil.which("systemctl") is None:
        return False
    if not os.getenv("XDG_RUNTIME_DIR") and os.getuid() != 0:  # type: ignore[attr-defined]
        # systemd --user usually requires XDG_RUNTIME_DIR; if it's missing the
        # call below would fail anyway, but root often runs without it.
        pass
    try:
        # ``is-system-running`` returns non-zero in some healthy states, so we
        # only care that systemctl was able to reach a user manager at all.
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    if result.returncode == 0:
        return True
    err = (result.stderr or "").lower()
    if "failed to connect" in err or "no such file or directory" in err:
        return False
    return False


def resolve_backend(requested: str) -> str:
    if requested in (None, "", AUTO_BACKEND):
        if in_container():
            return SUPERVISOR_BACKEND
        if systemd_user_available():
            return SYSTEMD_USER_BACKEND
        return SUPERVISOR_BACKEND
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported backend: {requested!r}. "
            f"choose from: auto/{'/'.join(SUPPORTED_BACKENDS)}"
        )
    return requested


@dataclass
class GuardServiceConfig:
    """Persistent configuration for a guard service installation."""

    backend: str = SUPERVISOR_BACKEND
    gpu_ids: list[int] = field(default_factory=list)
    idle_timeout: int = 5400
    placeholder_idle_s: float = 0.0
    placeholder_load: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)
    python_executable: str = ""
    gpulock_executable: str = ""

    def to_guard_argv(self) -> list[str]:
        argv: list[str] = ["guard"]
        for gid in self.gpu_ids:
            argv.append(str(gid))
        argv.extend(["--idle-timeout", str(self.idle_timeout)])
        argv.extend(["--placeholder-idle-s", str(self.placeholder_idle_s)])
        argv.append("--placeholder-load" if self.placeholder_load else "--no-placeholder-load")
        return argv

    def save(self, lock_root: Optional[Path] = None) -> Path:
        cfg_path = service_dir(lock_root) / "config.json"
        cfg_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        try:
            os.chmod(cfg_path, 0o600)
        except PermissionError:
            pass
        return cfg_path

    @classmethod
    def load(cls, lock_root: Optional[Path] = None) -> "GuardServiceConfig":
        cfg_path = service_dir(lock_root) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"no gpulock service config found at {cfg_path}. "
                "run `gpulock service install` first."
            )
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        gpu_ids = [int(x) for x in data.get("gpu_ids", [])]
        return cls(
            backend=str(data.get("backend", SUPERVISOR_BACKEND)),
            gpu_ids=gpu_ids,
            idle_timeout=int(data.get("idle_timeout", 5400)),
            placeholder_idle_s=float(data.get("placeholder_idle_s", 0.0)),
            placeholder_load=bool(data.get("placeholder_load", True)),
            extra_env={str(k): str(v) for k, v in dict(data.get("extra_env", {})).items()},
            python_executable=str(data.get("python_executable", "")),
            gpulock_executable=str(data.get("gpulock_executable", "")),
        )

    @classmethod
    def config_path(cls, lock_root: Optional[Path] = None) -> Path:
        return service_dir(lock_root) / "config.json"

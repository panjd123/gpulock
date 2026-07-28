from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def lock_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "lock-root"
    root.mkdir()
    monkeypatch.setenv("GPULOCK_LOCK_DIR", str(root))
    return root


@pytest.fixture
def cli_env(lock_root: Path, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GPULOCK_LOCK_DIR"] = str(lock_root)
    env["PYTHONPATH"] = str(REPO / "src")
    env["GPULOCK_AGENT_GLOBAL_PATHS"] = os.pathsep.join([
        str(tmp_path / "agent-home" / ".codex" / "AGENTS.md"),
        str(tmp_path / "agent-home" / ".trae" / "AGENTS.md"),
    ])
    return env


@pytest.fixture
def run_cli(cli_env: dict[str, str]):
    def _run(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "gpulock", *args],
            capture_output=True,
            text=True,
            env=cli_env,
            timeout=timeout,
        )

    return _run


@pytest.fixture(autouse=True)
def cleanup_pycache():
    yield
    for path in (REPO / "src").rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in (REPO / "tests").rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)

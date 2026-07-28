"""Self-update command for an editable gpulock checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .service import supervisor as supervisor_backend


_PREFIX = "[gpulock update]"


def _say(message: str) -> None:
    print(f"{_PREFIX} {message}")


def _warn(message: str) -> None:
    print(f"{_PREFIX} {message}", file=sys.stderr)


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    proc = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(detail or f"{start} is not inside a git repository")
    return Path(proc.stdout.strip()).resolve()


def _print_dirty_status(lines: list[str]) -> None:
    _warn("repository has uncommitted changes; refusing to update")
    for line in lines[:20]:
        print(f"  {line}", file=sys.stderr)
    if len(lines) > 20:
        print(f"  ... {len(lines) - 20} more", file=sys.stderr)


def _check_clean(repo: Path) -> bool:
    proc = _run_git(repo, ["status", "--porcelain"])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        _warn(f"failed to check git status in {repo}: {detail}")
        return False
    dirty = [line for line in proc.stdout.splitlines() if line.strip()]
    if dirty:
        _print_dirty_status(dirty)
        return False
    return True


def _pull(repo: Path) -> int:
    proc = _run_git(repo, ["pull", "--ff-only"])
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        _warn(f"git pull --ff-only failed in {repo}")
        return proc.returncode or 1
    return 0


def cmd_update(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="gpulock update",
        description="Pull the gpulock git checkout and restart the guard service if it was running.",
    )
    parser.parse_args(argv)

    try:
        repo = _git_repo_root()
    except RuntimeError as e:
        _warn(str(e))
        return 1

    if not _check_clean(repo):
        return 2

    was_running = supervisor_backend.running_pid() > 0
    _say(f"updating {repo}")
    rc = _pull(repo)
    if rc != 0:
        return rc

    if not was_running:
        _say("service was not running before update; leaving it stopped")
        return 0

    _say("service was running before update; restarting")
    rc = supervisor_backend.restart()
    if rc != 0:
        _warn(f"service restart failed with rc={rc}")
        return rc
    _say("update complete")
    return 0

from __future__ import annotations

import subprocess
from pathlib import Path

from gpulock import update


class _FakeSupervisor:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid
        self.restarts = 0

    def running_pid(self) -> int:
        return self.pid

    def restart(self) -> int:
        self.restarts += 1
        return 0


def _completed(args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_update_pulls_and_restarts_when_service_was_running(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    calls: list[list[str]] = []
    supervisor = _FakeSupervisor(pid=1234)

    monkeypatch.setattr(update, "_git_repo_root", lambda: repo)
    monkeypatch.setattr(update, "supervisor_backend", supervisor)

    def fake_run_git(_repo: Path, args: list[str]):
        calls.append(args)
        if args == ["status", "--porcelain"]:
            return _completed(args)
        if args == ["pull", "--ff-only"]:
            return _completed(args, stdout="Already up to date.\n")
        raise AssertionError(args)

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    rc = update.cmd_update([])

    assert rc == 0
    assert calls == [["status", "--porcelain"], ["pull", "--ff-only"]]
    assert supervisor.restarts == 1
    out = capsys.readouterr().out
    assert "Already up to date." in out
    assert "service was running before update; restarting" in out


def test_update_pulls_without_restart_when_service_was_stopped(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    supervisor = _FakeSupervisor(pid=0)

    monkeypatch.setattr(update, "_git_repo_root", lambda: repo)
    monkeypatch.setattr(update, "supervisor_backend", supervisor)

    def fake_run_git(_repo: Path, args: list[str]):
        if args == ["status", "--porcelain"]:
            return _completed(args)
        if args == ["pull", "--ff-only"]:
            return _completed(args, stdout="Already up to date.\n")
        raise AssertionError(args)

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    assert update.cmd_update([]) == 0
    assert supervisor.restarts == 0


def test_update_fails_on_dirty_repo_without_pull_or_restart(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    calls: list[list[str]] = []
    supervisor = _FakeSupervisor(pid=1234)

    monkeypatch.setattr(update, "_git_repo_root", lambda: repo)
    monkeypatch.setattr(update, "supervisor_backend", supervisor)

    def fake_run_git(_repo: Path, args: list[str]):
        calls.append(args)
        return _completed(args, stdout=" M src/gpulock/placeholder.py\n")

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    rc = update.cmd_update([])

    assert rc == 2
    assert calls == [["status", "--porcelain"]]
    assert supervisor.restarts == 0
    assert "repository has uncommitted changes" in capsys.readouterr().err


def test_update_fails_on_pull_error_without_restart(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    supervisor = _FakeSupervisor(pid=1234)

    monkeypatch.setattr(update, "_git_repo_root", lambda: repo)
    monkeypatch.setattr(update, "supervisor_backend", supervisor)

    def fake_run_git(_repo: Path, args: list[str]):
        if args == ["status", "--porcelain"]:
            return _completed(args)
        if args == ["pull", "--ff-only"]:
            return _completed(args, returncode=1, stderr="fatal: not possible to fast-forward\n")
        raise AssertionError(args)

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    rc = update.cmd_update([])

    assert rc == 1
    assert supervisor.restarts == 0
    err = capsys.readouterr().err
    assert "fatal: not possible to fast-forward" in err
    assert "git pull --ff-only failed" in err

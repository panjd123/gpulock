from __future__ import annotations

from importlib.resources import files

from gpulock.agent import (
    MARKER_END,
    MARKER_START,
    build_agent_output,
    load_agent_prompt,
)


POLICY_TITLE = "# GPU Execution Policy For Agents"


def test_agent_prompt_is_packaged_as_data():
    resource = files("gpulock").joinpath("data/GPULOCK_AGENT_PROMPT.md")
    assert resource.is_file()
    assert POLICY_TITLE in load_agent_prompt()


def test_agent_default_is_local(run_cli):
    proc = run_cli(["agent"])

    assert proc.returncode == 0, proc.stderr
    assert "LOCAL scope" in proc.stdout
    assert "./AGENTS.md" in proc.stdout
    assert POLICY_TITLE in proc.stdout
    assert MARKER_START in proc.stdout and MARKER_END in proc.stdout


def test_agent_local_flag_matches_default(run_cli):
    default = run_cli(["agent"])
    local = run_cli(["agent", "--local"])

    assert local.returncode == 0, local.stderr
    assert local.stdout == default.stdout


def test_agent_global_targets_tool_global_files(run_cli):
    proc = run_cli(["agent", "--global"])

    assert proc.returncode == 0, proc.stderr
    assert "GLOBAL scope" in proc.stdout
    assert "~/.codex/AGENTS.md" in proc.stdout
    assert "~/.trae/AGENTS.md" in proc.stdout
    assert POLICY_TITLE in proc.stdout


def test_agent_local_and_global_differ_only_in_preamble(run_cli):
    local = run_cli(["agent", "--local"]).stdout
    glob = run_cli(["agent", "--global"]).stdout

    assert local != glob
    # The policy block (between markers) is identical for both scopes.
    local_block = local.split(MARKER_START, 1)[1]
    glob_block = glob.split(MARKER_START, 1)[1]
    assert local_block == glob_block


def test_agent_help_lists_per_tool_install_commands(run_cli):
    proc = run_cli(["agent", "--help"])

    assert proc.returncode == 0, proc.stderr
    for snippet in (
        "codex exec --skip-git-repo-check",
        "coco -y -p",
        "agent -p -f",
        "claude -p --dangerously-skip-permissions",
    ):
        assert snippet in proc.stdout, snippet


def test_agent_rejects_combining_scopes(run_cli):
    proc = run_cli(["agent", "--local", "--global"])

    assert proc.returncode == 2
    assert "not allowed with" in proc.stderr or "usage:" in proc.stderr

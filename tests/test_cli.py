from __future__ import annotations

import json
import sys

from gpulock.cli import _parse_run_args


def test_check_command_injects_cuda_visible_devices(run_cli):
    proc = run_cli([
        "check",
        "99",
        "--",
        sys.executable,
        "-c",
        "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', '<missing>'))",
    ])

    assert proc.returncode == 0, proc.stderr
    assert "99" in proc.stdout.splitlines()


def test_run_flags_can_appear_before_or_after_gpu_ids():
    before = _parse_run_args([
        "perf",
        "--wait-gpu-idle",
        "0",
        "--",
        "echo",
        "ok",
    ])
    after = _parse_run_args([
        "perf",
        "0",
        "--wait-gpu-idle",
        "--",
        "echo",
        "ok",
    ])

    assert before.wait_gpu_idle is True
    assert after.wait_gpu_idle is True
    assert before.command == ["--", "echo", "ok"]
    assert after.command == ["--", "echo", "ok"]


def test_child_flags_after_separator_are_not_parsed_as_run_flags():
    args = _parse_run_args([
        "perf",
        "0",
        "--",
        "echo",
        "--wait-gpu-idle",
    ])

    assert args.wait_gpu_idle is False
    assert args.command == ["--", "echo", "--wait-gpu-idle"]


def test_multi_gpu_command_injects_comma_separated_devices(run_cli):
    proc = run_cli([
        "check",
        "98,99",
        "--",
        sys.executable,
        "-c",
        "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', '<missing>'))",
    ])

    assert proc.returncode == 0, proc.stderr
    assert "98,99" in proc.stdout


def test_default_command_mode_joins_tokens_into_quoted_shell_command(run_cli):
    payload = "space value 'single' \"double\" --literal $HOME *"
    proc = run_cli([
        "check",
        "99",
        "--timeout-s",
        "5",
        "--",
        sys.executable,
        "-c",
        "import json, sys; print(json.dumps(sys.argv[1:]))",
        payload,
        "--child-flag",
        "two words",
    ])

    assert proc.returncode == 0, proc.stderr
    argv = json.loads(proc.stdout.splitlines()[-2])
    assert argv == [payload, "--child-flag", "two words"]


def test_child_command_flags_after_separator_are_not_parsed_by_gpulock_in_default_shell_mode(run_cli):
    proc = run_cli([
        "check",
        "99",
        "--",
        sys.executable,
        "-c",
        "import json, sys; print(json.dumps(sys.argv[1:]))",
        "--timeout-s",
        "child-value",
        "--",
        "after-child-separator",
    ])

    assert proc.returncode == 0, proc.stderr
    argv = json.loads(proc.stdout.splitlines()[-2])
    assert argv == ["--timeout-s", "child-value", "--", "after-child-separator"]


def test_single_string_command_uses_shell_for_convenience(run_cli):
    proc = run_cli([
        "check",
        "99",
        "--",
        "printf '%s\\n' \"$GPULOCK_LOCK_MODE:$CUDA_VISIBLE_DEVICES\" | tr a-z A-Z",
    ])

    assert proc.returncode == 0, proc.stderr
    assert "READ:99" in proc.stdout.splitlines()


def test_read_alias_matches_check(run_cli):
    proc = run_cli([
        "read",
        "99",
        "--",
        sys.executable,
        "-c",
        "import os; print(os.environ['GPULOCK_LOCK_MODE'])",
    ])

    assert proc.returncode == 0, proc.stderr
    assert "read" in proc.stdout.splitlines()


def test_write_alias_matches_perf(run_cli):
    proc = run_cli([
        "write",
        "99",
        "--timeout-s",
        "1",
        "--",
        sys.executable,
        "-c",
        "import os; print(os.environ['GPULOCK_LOCK_MODE'])",
    ])

    assert proc.returncode in (0, 1, 124)
    if proc.returncode == 0:
        assert "write" in proc.stdout.splitlines()


def test_unsupported_top_level_mode_flag_is_rejected(run_cli):
    proc = run_cli([
        "--mode",
        "read",
        "99",
        "--",
        sys.executable,
        "-c",
        "print('unused')",
    ])

    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr or "usage:" in proc.stderr


def test_standalone_lock_command_is_not_supported(run_cli):
    proc = run_cli(["lock", "99"])

    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr

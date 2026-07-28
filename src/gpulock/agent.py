"""``gpulock agent`` -- print the packaged agent policy with install guidance.

The canonical policy text lives inside the package as data
(``gpulock/data/GPULOCK_AGENT_PROMPT.md``) so that an installed ``gpulock`` can
always locate it, regardless of the current working directory.

``gpulock agent`` prints a short, scope-specific preamble that tells a coding
agent where and how to install the policy, followed by the policy itself wrapped
in stable markers so repeated installs update in place instead of duplicating.
The intended use is to feed the output to a coding-agent CLI, e.g.
``codex "$(gpulock agent)"``.
"""

from __future__ import annotations

import argparse
import os
import re
from importlib.resources import files
from pathlib import Path

PROMPT_RESOURCE = "data/GPULOCK_AGENT_PROMPT.md"

MARKER_START = "<!-- gpulock:start -->"
MARKER_END = "<!-- gpulock:end -->"
_MARKED_BLOCK_RE = re.compile(
    rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
    re.DOTALL,
)

LOCAL = "local"
GLOBAL = "global"

DEFAULT_GLOBAL_AGENT_PATHS = (
    Path("~/.codex/AGENTS.md"),
    Path("~/.trae/AGENTS.md"),
)


def load_agent_prompt() -> str:
    """Return the packaged agent policy text."""
    resource = files("gpulock").joinpath(PROMPT_RESOURCE)
    return resource.read_text(encoding="utf-8")


_SHARED_HANDLING = """\
File handling rules:

- If the target file does not exist, create it (and any missing parent
  directories) and write the policy block below.
- If the target file already exists, read it first. If it already contains a
  block delimited by `{start}` and `{end}`, replace everything between those
  markers in place. Otherwise append the policy block to the end, separated by a
  blank line.
- Never duplicate the block, and preserve all other existing content.
- Copy the policy block verbatim, including the marker comments.

When you are done, stop; do not run any other commands.""".format(
    start=MARKER_START, end=MARKER_END
)

_LOCAL_TARGET = """\
Target (LOCAL scope): `./AGENTS.md` -- the AGENTS.md in the current working
directory. This scopes the policy to the current project/checkout only."""

_GLOBAL_TARGET = """\
Target (GLOBAL scope): the global agent-instruction file of the coding-agent CLI
you are running as, so the policy applies to every project on this machine.
Choose the path that matches your tool:

- Codex CLI:       `~/.codex/AGENTS.md`
- Coco / Trae CLI: `~/.trae/AGENTS.md`
- Claude Code:     `~/.claude/CLAUDE.md`
- Cursor CLI:      no machine-global instruction file exists. Fall back to
                   `./AGENTS.md` in the current repo, or add the policy as a User
                   Rule in Cursor settings.

If you are unsure which tool you are, prefer the one whose config directory
already exists under `$HOME`."""


def _preamble(scope: str) -> str:
    target = _GLOBAL_TARGET if scope == GLOBAL else _LOCAL_TARGET
    return (
        "# gpulock: install GPU execution policy\n\n"
        "You are an AI coding agent. The block delimited by the markers below is a "
        "GPU execution policy for this environment. Install it into the target file "
        "described here so that future sessions pick it up automatically.\n\n"
        f"{target}\n\n"
        f"{_SHARED_HANDLING}\n"
    )


def build_agent_output(scope: str) -> str:
    """Build the full ``gpulock agent`` output for the given scope."""
    policy = load_agent_prompt().strip("\n")
    return (
        f"{_preamble(scope)}\n"
        f"{MARKER_START}\n"
        f"{policy}\n"
        f"{MARKER_END}\n"
    )


def build_policy_block() -> str:
    """Return only the marked policy block suitable for direct file writes."""
    policy = load_agent_prompt().strip("\n")
    return f"{MARKER_START}\n{policy}\n{MARKER_END}\n"


def install_policy_block(path: Path, block: str | None = None) -> bool:
    """Create or update ``path`` with the marked gpulock policy block.

    Returns ``True`` when the file content changed. Existing content outside the
    marker block is preserved, and repeated calls update the block in place
    rather than appending duplicates.
    """
    block = build_policy_block() if block is None else block
    try:
        original = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        original = ""

    if _MARKED_BLOCK_RE.search(original):
        updated = _MARKED_BLOCK_RE.sub(block.rstrip("\n"), original, count=1)
    elif original.strip():
        updated = f"{original.rstrip()}\n\n{block}"
    else:
        updated = block

    if updated == original:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return True


def global_agent_paths() -> list[Path]:
    """Return global AGENTS.md targets for common coding-agent CLIs."""
    override = os.getenv("GPULOCK_AGENT_GLOBAL_PATHS", "").strip()
    if override:
        return [Path(item).expanduser() for item in override.split(os.pathsep) if item]
    return [path.expanduser() for path in DEFAULT_GLOBAL_AGENT_PATHS]


def install_global_agent_policies(paths: list[Path] | None = None) -> list[tuple[Path, bool]]:
    """Install the gpulock policy block into all global AGENTS.md targets."""
    block = build_policy_block()
    targets = global_agent_paths() if paths is None else paths
    return [(path, install_policy_block(path, block)) for path in targets]


INSTALL_HELP = """\
Install the policy by feeding this command's output to a coding agent. Pick the
line that matches your tool:

  Codex CLI (non-interactive; -p means --profile, not print):
    codex exec --skip-git-repo-check "$(gpulock agent)"           # ./AGENTS.md
    codex exec --skip-git-repo-check "$(gpulock agent --global)"  # ~/.codex/AGENTS.md

  Coco / Trae CLI (-y auto-approves the edit):
    coco -y -p "$(gpulock agent --global)"                        # ~/.trae/AGENTS.md

  Cursor CLI (command is `agent`; -f allows the write; no machine-global file):
    agent -p -f "$(gpulock agent --local)"                        # ./AGENTS.md

  Claude Code (--dangerously-skip-permissions allows the write):
    claude -p --dangerously-skip-permissions "$(gpulock agent --global)" </dev/null

--global targets the tool's global AGENTS.md; --local (default) targets
./AGENTS.md. Re-running updates the existing gpulock block in place.
"""


def cmd_agent(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="gpulock agent",
        description=(
            "Print the gpulock agent policy plus instructions for installing it "
            "into an AGENTS.md file."
        ),
        epilog=INSTALL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--local",
        dest="scope",
        action="store_const",
        const=LOCAL,
        help="Install guidance targets ./AGENTS.md in the current directory (default).",
    )
    scope.add_argument(
        "--global",
        dest="scope",
        action="store_const",
        const=GLOBAL,
        help="Install guidance targets the coding-agent tool's global AGENTS.md.",
    )
    parser.set_defaults(scope=LOCAL)
    args = parser.parse_args(argv)

    print(build_agent_output(args.scope), end="")
    return 0

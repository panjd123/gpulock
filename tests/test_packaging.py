from __future__ import annotations

import tomllib

from conftest import REPO


def test_pyproject_declares_current_entrypoint_and_dependencies():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())

    assert pyproject["project"]["description"] == "GPU read/write lock wrapper with an idle guard service"
    assert pyproject["project"]["keywords"] == ["gpu", "lock", "cuda", "nvidia", "supervisor"]
    assert pyproject["project"]["dependencies"] == [
        "setproctitle>=1.3; platform_system == 'Linux'",
        "supervisor>=4.2",
        "torch",
    ]
    assert pyproject["project"]["scripts"] == {
        "gpulock": "gpulock.cli:main",
    }
    assert pyproject["project"]["optional-dependencies"]["test"] == ["pytest>=8"]


def test_install_script_is_not_part_of_the_project():
    assert not (REPO / "install.sh").exists()

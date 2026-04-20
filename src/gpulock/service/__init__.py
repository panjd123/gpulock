"""``gpulock service``: install/manage the guard daemon as a long-running service.

This subpackage provides two backends:

* ``systemd-user`` -- writes a ``~/.config/systemd/user/gpulock-guard.service``
  unit file and uses ``systemctl --user`` for everything.
* ``supervisor`` -- a built-in mini supervisor for environments without
  systemd (Docker containers, minimal images...). It daemonizes itself,
  spawns ``gpulock guard`` as a child, and restarts it on crash.

Backend selection (``--backend auto``):

1. Container is detected (``/.dockerenv`` exists or PID 1 cgroup mentions
   ``docker``/``containerd``/``kubepods``/``crio``) -> ``supervisor``.
2. ``systemctl --user`` is usable -> ``systemd-user``.
3. Otherwise -> ``supervisor``.
"""

from __future__ import annotations

from .cli import cmd_service

__all__ = ["cmd_service"]

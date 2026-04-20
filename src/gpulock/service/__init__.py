"""``gpulock service``: install/manage the guard daemon as a long-running service.

Single backend: the third-party `supervisor` package (a.k.a. supervisord) is
used to daemonize and supervise ``gpulock guard`` everywhere (bare metal and
containers alike). See ``supervisor.py`` for the integration.
"""

from __future__ import annotations

from .cli import cmd_service

__all__ = ["cmd_service"]

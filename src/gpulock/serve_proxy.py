"""Reverse-proxy serve mode for gpulock.

Instead of monkey-patching the inference server (vLLM/SGLang/...), ``gpulock
serve <listen>:<backend> <gpus> -- <cmd>`` runs a tiny HTTP reverse proxy in
front of the unmodified backend server. The proxy counts in-flight *real*
requests (heartbeat/probe endpoints are filtered out via a blacklist) and drives
the ``serve.busy`` signal file directly, in-process:

    - real request 0 -> 1 : touch serve.busy   -> guard parks the placeholder
    - real request 1 -> 0 : (after debounce) clear serve.busy -> guard activates

This is framework-agnostic: the backend server runs with its own native CLI and
is never patched. The only cost is a local 127.0.0.1 forwarding hop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import threading
import time
from typing import Callable, Iterable

logger = logging.getLogger("gpulock.serve_proxy")

# Hosts that mean "listen on every interface". For these we bind both IPv4 and
# IPv6 so clients can reach the proxy over either stack.
_WILDCARD_HOSTS = frozenset({"", "*", "0.0.0.0", "::"})

# ---------------------------------------------------------------------------
# Request blacklist (heartbeat / probe / metadata endpoints)
# ---------------------------------------------------------------------------

# (method, path) rules. ``path`` matches exactly or as a prefix (so /v1/models
# also covers /v1/models/<id>). ``method == "*"`` matches any method.
# Anything NOT matched here is treated as a real request and triggers active.
DEFAULT_IGNORE_RULES: tuple[tuple[str, str], ...] = (
    # liveness / readiness probes
    ("GET", "/health"),
    ("GET", "/healthz"),
    ("GET", "/health_generate"),  # SGLang
    ("GET", "/ready"),
    ("GET", "/ping"),
    # metrics / load / stats scrapers
    ("GET", "/metrics"),
    ("GET", "/stats"),
    ("GET", "/load"),  # SGLang
    # metadata / discovery (typical client startup probes)
    ("GET", "/version"),
    ("GET", "/get_model_info"),  # SGLang
    ("GET", "/get_server_info"),  # SGLang
    ("GET", "/v1/models"),  # also /v1/models/<id>
    # CORS preflight is never a real inference request
    ("*", "OPTIONS_PREFLIGHT"),
)


def parse_ignore_rules(
    raw_items: Iterable[str] | None,
    *,
    reset: bool = False,
    base: tuple[tuple[str, str], ...] = DEFAULT_IGNORE_RULES,
) -> list[tuple[str, str]]:
    """Build the ignore-rule list from the defaults plus user additions.

    Each raw item is ``METHOD:path`` or a bare ``path`` (bare paths default to
    the GET method). With ``reset=True`` the defaults are dropped and only the
    user-provided rules are used.
    """
    rules: list[tuple[str, str]] = [] if reset else list(base)
    for item in raw_items or []:
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            method, path = token.split(":", 1)
            method = method.strip().upper() or "*"
            path = path.strip()
        else:
            method, path = "GET", token
        if not path.startswith("/"):
            path = "/" + path
        rules.append((method, path))
    return rules


def ignore_rules_from_env(
    cli_items: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve ignore rules from CLI args + ``GPULOCK_SERVE_IGNORE`` env.

    ``GPULOCK_SERVE_IGNORE`` is a comma-separated list of ``METHOD:path`` or
    bare ``path`` entries. ``GPULOCK_SERVE_IGNORE_RESET=1`` drops the built-in
    defaults so only env/CLI rules apply.
    """
    reset = os.environ.get("GPULOCK_SERVE_IGNORE_RESET", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    env_raw = os.environ.get("GPULOCK_SERVE_IGNORE", "")
    items: list[str] = []
    for chunk in env_raw.split(","):
        if chunk.strip():
            items.append(chunk.strip())
    if cli_items:
        items.extend(cli_items)
    return parse_ignore_rules(items, reset=reset)


def should_count(
    method: str,
    path: str,
    rules: Iterable[tuple[str, str]] = DEFAULT_IGNORE_RULES,
) -> bool:
    """Return True if a request should count as real activity (trigger active).

    Heartbeat/probe requests matched by the blacklist return False. Everything
    else (chat/completions/embeddings/generate/...) returns True.
    """
    method = (method or "").upper()
    # Normalize the path: strip query string, drop trailing slash (except root).
    clean = path.split("?", 1)[0]
    if len(clean) > 1:
        clean = clean.rstrip("/")
    for rule_method, rule_path in rules:
        if rule_path == "OPTIONS_PREFLIGHT":
            if method == "OPTIONS":
                return False
            continue
        if rule_method != "*" and rule_method != method:
            continue
        if clean == rule_path or clean.startswith(rule_path + "/"):
            return False
    return True


# ---------------------------------------------------------------------------
# Request counter with debounced serve.busy management
# ---------------------------------------------------------------------------

class RequestCounter:
    """Thread/loop-safe in-flight counter that drives the serve.busy signal.

    ``on_busy``/``on_idle`` are zero-arg callbacks that set/clear the signal
    file (wired to gpulock's _touch_serve_signal/_clear_serve_signal). The idle
    transition is debounced so a burst of finishing requests doesn't flap the
    signal off/on repeatedly.
    """

    def __init__(
        self,
        on_busy: Callable[[], None],
        on_idle: Callable[[], None],
        *,
        debounce_s: float = 0.05,
    ) -> None:
        self._on_busy = on_busy
        self._on_idle = on_idle
        self._debounce_s = max(float(debounce_s), 0.0)
        self._count = 0
        self._lock = threading.Lock()
        self._idle_token = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def _set_busy(self) -> None:
        try:
            self._on_busy()
        except Exception as e:  # noqa: BLE001
            logger.debug("serve.busy set failed: %s", e)

    def _set_idle(self) -> None:
        try:
            self._on_idle()
        except Exception as e:  # noqa: BLE001
            logger.debug("serve.busy clear failed: %s", e)

    def increment(self) -> None:
        with self._lock:
            self._count += 1
            self._idle_token += 1  # cancel any pending idle check
            first = self._count == 1
        if first:
            self._set_busy()

    def decrement(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1
            if self._count != 0:
                return
            self._idle_token += 1
            token = self._idle_token
        self._schedule_idle(token)

    def _schedule_idle(self, token: int) -> None:
        if self._debounce_s <= 0.0:
            self._maybe_idle(token)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            threading.Timer(self._debounce_s, self._maybe_idle, args=(token,)).start()
            return
        loop.call_later(self._debounce_s, self._maybe_idle, token)

    def _maybe_idle(self, token: int) -> None:
        with self._lock:
            if token != self._idle_token or self._count != 0:
                return
        self._set_idle()


# ---------------------------------------------------------------------------
# aiohttp reverse proxy
# ---------------------------------------------------------------------------

# Hop-by-hop headers must not be forwarded (RFC 7230 6.1).
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    )
)


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _make_ipv4_first_connector():
    """Build a TCPConnector whose resolver returns IPv4 addresses first.

    Internal traffic to the backend should prefer IPv4 (vLLM/SGLang bind
    127.0.0.1 by default), but we still fall back to IPv6 so an IPv6-only or
    ``[::1]`` backend keeps working. Literal IP backends are unaffected: their
    single family is returned as-is.
    """
    from aiohttp import TCPConnector
    from aiohttp.resolver import DefaultResolver

    class _IPv4FirstResolver(DefaultResolver):
        async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
            # Resolve over both families, then order IPv4 before IPv6.
            results = await super().resolve(host, port, socket.AF_UNSPEC)
            results.sort(key=lambda r: 0 if r["family"] == socket.AF_INET else 1)
            return results

    return TCPConnector(resolver=_IPv4FirstResolver(), family=socket.AF_UNSPEC)


def build_proxy_app(
    backend_url: str,
    counter: RequestCounter,
    rules: Iterable[tuple[str, str]],
):
    """Build an aiohttp Application that reverse-proxies to ``backend_url``."""
    from aiohttp import ClientSession, ClientTimeout, web

    rules = list(rules)
    app = web.Application()

    async def _on_startup(app: "web.Application") -> None:
        # No total timeout: streaming generations can run for minutes.
        # Prefer IPv4 when talking to the local backend (internal traffic is
        # IPv4-first), but keep IPv6 as a fallback.
        app["session"] = ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=30),
            connector=_make_ipv4_first_connector(),
        )

    async def _on_cleanup(app: "web.Application") -> None:
        session = app.get("session")
        if session is not None:
            await session.close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    async def handler(request: "web.Request") -> "web.StreamResponse":
        session: ClientSession = request.app["session"]
        target = backend_url + request.rel_url.raw_path_qs
        counted = should_count(request.method, request.rel_url.raw_path)
        if counted:
            counter.increment()
        try:
            async with session.request(
                request.method,
                target,
                headers=_filter_headers(request.headers),
                data=request.content,  # stream the request body through
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers=_filter_headers(upstream.headers),
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("proxy error for %s %s: %s", request.method, target, e)
            return web.Response(status=502, text=f"gpulock serve proxy: backend error: {e}")
        finally:
            if counted:
                counter.decrement()

    app.router.add_route("*", "/{tail:.*}", handler)
    return app


async def _wait_backend_ready(backend_url: str, timeout_s: float | None = None) -> bool:
    """Wait until the backend TCP port accepts connections.

    Returns True when the backend is reachable and False when the timeout is
    reached. Callers decide whether a timeout should fail the serve command or
    start proxying anyway.
    """
    from urllib.parse import urlparse

    parsed = urlparse(backend_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    deadline = None if timeout_s is None or timeout_s <= 0 else time.monotonic() + timeout_s
    while deadline is None or time.monotonic() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            with __import__("contextlib").suppress(Exception):
                await writer.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(0.5)
    logger.warning("backend %s not reachable after %.0fs", backend_url, timeout_s or 0)
    return False


def _format_host_for_url(host: str) -> str:
    """Wrap IPv6 literals in brackets so they are valid inside a URL authority."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _make_listen_sockets(listen_host: str, listen_port: int) -> list[socket.socket]:
    """Create bound listen sockets for ``listen_host``.

    A wildcard host (``0.0.0.0``/``::``/``*``/empty) binds *both* IPv4 and IPv6
    so clients can connect over either stack. The IPv6 socket is set
    ``IPV6_V6ONLY`` so the two sockets can share the same port without the
    "address already in use" clash that a dual-stack IPv6 socket would cause.

    A specific host binds only the family that host resolves to.
    """
    socks: list[socket.socket] = []
    if listen_host in _WILDCARD_HOSTS:
        specs = [(socket.AF_INET, "0.0.0.0")]
        if socket.has_ipv6:
            specs.append((socket.AF_INET6, "::"))
    else:
        # Resolve the explicit host to its concrete family/families.
        infos = socket.getaddrinfo(
            listen_host, listen_port, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        seen: set[tuple[int, str]] = set()
        specs = []
        for family, _type, _proto, _canon, sockaddr in infos:
            key = (family, sockaddr[0])
            if key in seen:
                continue
            seen.add(key)
            specs.append((family, sockaddr[0]))

    for family, addr in specs:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            # Keep IPv4 and IPv6 wildcard sockets independent on the same port.
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            sock.bind((addr, listen_port))
        except OSError as e:
            sock.close()
            # Tolerate a missing family (e.g. IPv6 disabled) when we asked for a
            # wildcard dual-stack bind, but only if at least one socket survives.
            if listen_host in _WILDCARD_HOSTS and socks:
                logger.warning("skip listen on %s:%d: %s", addr, listen_port, e)
                continue
            for s in socks:
                s.close()
            raise
        sock.listen(128)
        sock.setblocking(False)
        socks.append(sock)
    return socks


async def run_proxy(
    listen_host: str,
    listen_port: int,
    backend_host: str,
    backend_port: int,
    counter: RequestCounter,
    rules: Iterable[tuple[str, str]],
    *,
    ready_event: asyncio.Event | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the reverse proxy until ``stop_event`` is set."""
    from aiohttp import web

    backend_url = f"http://{_format_host_for_url(backend_host)}:{backend_port}"
    app = build_proxy_app(backend_url, counter, rules)
    runner = web.AppRunner(app)
    await runner.setup()

    sockets = _make_listen_sockets(listen_host, listen_port)
    for sock in sockets:
        site = web.SockSite(runner, sock)
        await site.start()
    bound = ", ".join(
        f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in sockets
    )
    logger.info("serve proxy listening on %s -> %s", bound, backend_url)
    if ready_event is not None:
        ready_event.set()
    try:
        if stop_event is not None:
            await stop_event.wait()
        else:
            while True:
                await asyncio.sleep(3600)
    finally:
        await runner.cleanup()

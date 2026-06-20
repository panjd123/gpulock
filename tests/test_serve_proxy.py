from __future__ import annotations

import asyncio
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gpulock import serve_proxy as sp


# ---------------------------------------------------------------------------
# Blacklist / should_count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method,path,expected",
    [
        ("POST", "/v1/chat/completions", True),
        ("POST", "/v1/completions", True),
        ("POST", "/v1/embeddings", True),
        ("POST", "/generate", True),
        ("GET", "/health", False),
        ("GET", "/healthz", False),
        ("GET", "/metrics", False),
        ("GET", "/metrics?foo=1", False),
        ("GET", "/v1/models", False),
        ("GET", "/v1/models/deepseek-v4", False),  # prefix match
        ("GET", "/get_model_info", False),  # SGLang
        ("OPTIONS", "/v1/chat/completions", False),  # CORS preflight
        ("GET", "/v1/models/", False),  # trailing slash normalized
    ],
)
def test_should_count_default_rules(method, path, expected):
    assert sp.should_count(method, path, sp.DEFAULT_IGNORE_RULES) is expected


def test_parse_ignore_rules_adds_to_defaults():
    rules = sp.parse_ignore_rules(["/v1/foo", "POST:/bar"])
    # defaults still present
    assert sp.should_count("GET", "/health", rules) is False
    # new bare path (defaults to GET) ignored
    assert sp.should_count("GET", "/v1/foo", rules) is False
    # POST:/bar ignored for POST but counted for GET
    assert sp.should_count("POST", "/bar", rules) is False
    assert sp.should_count("GET", "/bar", rules) is True


def test_parse_ignore_rules_reset_drops_defaults():
    rules = sp.parse_ignore_rules(["/only-this"], reset=True)
    # default health is now counted (no longer blacklisted)
    assert sp.should_count("GET", "/health", rules) is True
    assert sp.should_count("GET", "/only-this", rules) is False


def test_ignore_rules_from_env(monkeypatch):
    monkeypatch.setenv("GPULOCK_SERVE_IGNORE", "/from-env, POST:/x")
    rules = sp.ignore_rules_from_env(["/from-cli"])
    assert sp.should_count("GET", "/from-env", rules) is False
    assert sp.should_count("POST", "/x", rules) is False
    assert sp.should_count("GET", "/from-cli", rules) is False
    assert sp.should_count("GET", "/health", rules) is False  # defaults kept


def test_ignore_rules_from_env_reset(monkeypatch):
    monkeypatch.setenv("GPULOCK_SERVE_IGNORE", "/keep")
    monkeypatch.setenv("GPULOCK_SERVE_IGNORE_RESET", "1")
    rules = sp.ignore_rules_from_env()
    assert sp.should_count("GET", "/keep", rules) is False
    assert sp.should_count("GET", "/health", rules) is True  # defaults dropped


# ---------------------------------------------------------------------------
# RequestCounter signal transitions
# ---------------------------------------------------------------------------

def test_request_counter_busy_idle_transitions():
    events: list[str] = []
    counter = sp.RequestCounter(
        on_busy=lambda: events.append("busy"),
        on_idle=lambda: events.append("idle"),
        debounce_s=0.0,  # synchronous idle for deterministic test
    )
    counter.increment()  # 0 -> 1
    counter.increment()  # 1 -> 2 (no extra busy)
    counter.decrement()  # 2 -> 1 (no idle)
    assert events == ["busy"]
    counter.decrement()  # 1 -> 0 -> idle
    assert events == ["busy", "idle"]


def test_request_counter_debounce_cancelled_by_new_request():
    events: list[str] = []
    counter = sp.RequestCounter(
        on_busy=lambda: events.append("busy"),
        on_idle=lambda: events.append("idle"),
        debounce_s=0.1,
    )

    async def scenario():
        counter.increment()
        counter.decrement()  # schedules idle in 0.1s
        await asyncio.sleep(0.02)
        counter.increment()  # cancels the pending idle
        await asyncio.sleep(0.15)
        counter.decrement()  # schedules idle again
        await asyncio.sleep(0.15)

    asyncio.run(scenario())
    # busy fires once at start, busy again is NOT fired (count went 0->1 twice
    # but the second increment also crosses 0->1 so busy fires twice total),
    # idle fires once at the very end.
    assert events.count("idle") == 1
    assert events[-1] == "idle"


# ---------------------------------------------------------------------------
# End-to-end reverse proxy against a fake backend
# ---------------------------------------------------------------------------

class _FakeBackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _respond(self):
        if self.path.startswith("/health") or self.path.startswith("/v1/models"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: chunk{i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            return
        # default echo
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"hello")

    def do_GET(self):
        self._respond()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._respond()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def fake_backend():
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _FakeBackendHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield port
    server.shutdown()


def test_proxy_forwards_and_signals(fake_backend):
    import aiohttp

    listen_port = _free_port()
    signals: list[str] = []
    counter = sp.RequestCounter(
        on_busy=lambda: signals.append("on"),
        on_idle=lambda: signals.append("off"),
        debounce_s=0.05,
    )
    rules = sp.DEFAULT_IGNORE_RULES

    async def scenario():
        stop = asyncio.Event()
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            sp.run_proxy(
                "127.0.0.1", listen_port,
                "127.0.0.1", fake_backend,
                counter, rules,
                ready_event=ready, stop_event=stop,
            )
        )
        await ready.wait()
        base = f"http://127.0.0.1:{listen_port}"
        async with aiohttp.ClientSession() as s:
            # heartbeat: must NOT trigger a signal
            async with s.get(base + "/health") as r:
                assert r.status == 200
                assert await r.read() == b"ok"
            async with s.get(base + "/v1/models") as r:
                assert r.status == 200
            await asyncio.sleep(0.15)
            assert signals == [], "heartbeat requests must not toggle serve.busy"

            # real request: triggers on, then off after debounce
            async with s.post(base + "/v1/chat/completions", json={"x": 1}) as r:
                assert r.status == 200
                assert await r.read() == b"hello"
            await asyncio.sleep(0.2)
            assert "on" in signals
            assert signals[-1] == "off"

            # streaming response is forwarded chunk by chunk and complete
            async with s.get(base + "/stream") as r:
                body = await r.read()
            assert b"chunk0" in body and b"chunk1" in body and b"chunk2" in body

        stop.set()
        await proxy_task

    asyncio.run(scenario())
    assert counter.count == 0

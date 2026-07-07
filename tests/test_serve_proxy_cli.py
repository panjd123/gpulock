from __future__ import annotations

import socket
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from gpulock.cli import _LISTEN_BACKEND_RE, _parse_proxy_spec, _parse_serve_proxy_args


def test_listen_backend_regex_matches_proxy_spec():
    assert _LISTEN_BACKEND_RE.match("8000:8001")
    assert _LISTEN_BACKEND_RE.match("0.0.0.0:8000:8001")
    # plain gpu-id forms must NOT look like a proxy spec
    assert not _LISTEN_BACKEND_RE.match("2,3")
    assert not _LISTEN_BACKEND_RE.match("2")
    assert not _LISTEN_BACKEND_RE.match("8000")


def test_parse_proxy_spec_host_combinations():
    # no host on either side
    assert _parse_proxy_spec("8000:8001") == ("0.0.0.0", 8000, "127.0.0.1", 8001)
    # listen host only
    assert _parse_proxy_spec("127.0.0.1:8000:8001") == ("127.0.0.1", 8000, "127.0.0.1", 8001)
    # backend host only
    assert _parse_proxy_spec("8000:127.0.0.1:8001") == ("0.0.0.0", 8000, "127.0.0.1", 8001)
    # both hosts
    assert _parse_proxy_spec("0.0.0.0:8000:127.0.0.1:8001") == ("0.0.0.0", 8000, "127.0.0.1", 8001)
    # hostnames work too
    assert _parse_proxy_spec("localhost:8000:localhost:8001") == ("localhost", 8000, "localhost", 8001)
    # gpu-id strings are not proxy specs
    assert _parse_proxy_spec("2,3") is None
    assert _parse_proxy_spec("2") is None


def test_parse_proxy_spec_ipv6_literals():
    # IPv6 wildcard listen, IPv4 backend (brackets stripped from the result)
    assert _parse_proxy_spec("[::]:8000:8001") == ("::", 8000, "127.0.0.1", 8001)
    # IPv6 on both sides
    assert _parse_proxy_spec("[::]:8000:[::1]:8001") == ("::", 8000, "::1", 8001)
    # IPv4 listen, IPv6 backend
    assert _parse_proxy_spec("0.0.0.0:8000:[::1]:8001") == ("0.0.0.0", 8000, "::1", 8001)
    # full IPv6 literal address
    assert _parse_proxy_spec("[2001:db8::1]:8000:8001") == (
        "2001:db8::1",
        8000,
        "127.0.0.1",
        8001,
    )
    # bare (unbracketed) IPv6 is rejected: ambiguous against the separators
    assert _parse_proxy_spec("::1:8000:8001") is None


def test_parse_serve_proxy_args_splits_command_and_options():
    args = _parse_serve_proxy_args(
        [
            "8000:8001",
            "2,3",
            "--debounce-ms",
            "20",
            "--backend-ready-timeout-s",
            "123",
            "--backend-ready-timeout-action",
            "proxy",
            "--",
            "echo",
            "hi",
        ]
    )
    assert args.listen_backend == "8000:8001"
    assert args.gpu_ids == "2,3"
    assert args.debounce_ms == 20
    assert args.backend_ready_timeout_s == 123
    assert args.backend_ready_timeout_action == "proxy"
    assert args.command == ["--", "echo", "hi"]


def test_parse_serve_proxy_args_defaults_to_no_backend_ready_timeout():
    args = _parse_serve_proxy_args(["8000:8001", "2,3", "--", "echo", "hi"])
    assert args.backend_ready_timeout_s is None
    assert args.no_backend_ready_timeout is False
    assert args.backend_ready_timeout_action == "fail"


def test_park_placeholder_until_ready_default_and_override(monkeypatch):
    # Enabled by default.
    args = _parse_serve_proxy_args(["8000:8001", "0", "--", "echo", "hi"])
    assert args.park_placeholder_until_ready is True
    # Explicit opt-out.
    args = _parse_serve_proxy_args(
        ["8000:8001", "0", "--no-park-placeholder-until-ready", "--", "echo", "hi"]
    )
    assert args.park_placeholder_until_ready is False
    # Env opt-out is honored as the default when the flag is omitted.
    monkeypatch.setenv("GPULOCK_SERVE_PARK_UNTIL_READY", "0")
    args = _parse_serve_proxy_args(["8000:8001", "0", "--", "echo", "hi"])
    assert args.park_placeholder_until_ready is False


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b"backend-ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.do_GET()


def test_serve_proxy_cli_end_to_end(run_cli, lock_root: Path):
    """`gpulock serve <listen>:<backend> <gpu> -- <cmd>` proxies and manages markers."""
    backend_port = _free_port()
    listen_port = _free_port()

    server = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()

    gpu_id = 97
    managed = lock_root / f"gpu{gpu_id}" / "serve.managed"

    # Run the proxy in a background thread; it blocks until the backend exits.
    # We use `sleep` as the backend "command" so the proxy stays up, then poll.
    import subprocess
    import sys

    env = {
        **__import__("os").environ,
        "GPULOCK_LOCK_DIR": str(lock_root),
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src"),
    }
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "gpulock", "serve",
            f"127.0.0.1:{listen_port}:{backend_port}", str(gpu_id),
            "--no-wait-gpu-idle",
            "--", "sleep", "10",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for the proxy to come up and mark the GPU as serve-managed.
        deadline = time.time() + 15
        while time.time() < deadline and not managed.exists():
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise AssertionError(f"proxy exited early: {out}\n{err}")
            time.sleep(0.1)
        assert managed.exists(), "serve.managed marker should be created"

        # A real request is proxied to the backend.
        url = f"http://127.0.0.1:{listen_port}/v1/chat/completions"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.time() + 10
        body = None
        while time.time() < deadline:
            try:
                with opener.open(url, data=b"{}", timeout=2) as r:
                    body = r.read()
                break
            except Exception:
                time.sleep(0.2)
        assert body == b"backend-ok", "proxy must forward to backend"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        server.shutdown()

    # After shutdown the markers are cleared.
    assert not managed.exists(), "serve.managed must be cleared on exit"
    assert not (lock_root / f"gpu{gpu_id}" / "serve.busy").exists()


def test_serve_busy_asserted_until_backend_ready(run_cli, lock_root: Path):
    """serve.busy must be held from launch until the backend is reachable.

    This keeps the guard's placeholder parked during the backend's
    compile/warmup/autotune window (the whole reason this option exists), then
    releases it to request-driven control once the backend is ready.
    """
    import subprocess
    import sys

    backend_port = _free_port()
    listen_port = _free_port()
    gpu_id = 96
    gpu_dir = lock_root / f"gpu{gpu_id}"
    managed = gpu_dir / "serve.managed"
    busy = gpu_dir / "serve.busy"
    startup = gpu_dir / "serve.startup"

    env = {
        **__import__("os").environ,
        "GPULOCK_LOCK_DIR": str(lock_root),
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src"),
    }
    # Backend deliberately does NOT listen yet; we start it later to create a
    # real "not ready" window.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "gpulock", "serve",
            f"127.0.0.1:{listen_port}:{backend_port}", str(gpu_id),
            "--no-wait-gpu-idle",
            "--", "sleep", "20",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = None
    try:
        # Once managed, serve.busy must already be asserted (backend not ready).
        deadline = time.time() + 15
        while time.time() < deadline and not managed.exists():
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise AssertionError(f"proxy exited early: {out}\n{err}")
            time.sleep(0.05)
        assert managed.exists(), "serve.managed should be created"
        assert busy.exists(), "serve.busy must be asserted before backend ready"
        assert startup.exists(), "serve.startup must be set before backend ready"

        # It must STAY asserted while the backend stays down.
        time.sleep(1.0)
        assert busy.exists(), "serve.busy must remain while backend is not ready"
        assert startup.exists(), "serve.startup must remain while backend is not ready"

        # Now bring the backend up; the hold must be released after readiness.
        server = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
        Thread(target=server.serve_forever, daemon=True).start()

        deadline = time.time() + 15
        while time.time() < deadline and (busy.exists() or startup.exists()):
            time.sleep(0.05)
        assert not busy.exists(), "serve.busy must clear once backend is ready"
        assert not startup.exists(), "serve.startup must clear once backend is ready"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if server is not None:
            server.shutdown()

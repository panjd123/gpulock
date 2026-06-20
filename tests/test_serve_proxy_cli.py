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
        ["8000:8001", "2,3", "--debounce-ms", "20", "--", "echo", "hi"]
    )
    assert args.listen_backend == "8000:8001"
    assert args.gpu_ids == "2,3"
    assert args.debounce_ms == 20
    assert args.command == ["--", "echo", "hi"]


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
        deadline = time.time() + 10
        body = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, data=b"{}", timeout=2) as r:
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

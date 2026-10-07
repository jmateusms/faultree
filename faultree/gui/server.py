"""Local web server of the faultree GUI (standard library only).

``faultree gui`` starts a :class:`http.server.ThreadingHTTPServer` on
127.0.0.1 that serves the static page in ``faultree/gui/static`` and a few
JSON endpoints backed by :mod:`faultree.gui.engine`. Each request runs in its
own thread, so a long analysis does not block the page or other requests.
Nothing is loaded from the network: the page works offline.
"""
from __future__ import annotations

import errno
import json
import mimetypes
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import unquote, urlsplit

from ..builder import _faultree_version
from . import engine

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8765
MAX_BODY = 32 * 1024 * 1024
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def find_static(name: str) -> Optional[Path]:
    """A file inside ``static/`` (never outside it)."""
    target = (STATIC_DIR / name).resolve()
    if STATIC_DIR in target.parents and target.is_file():
        return target
    return None


def _post_routes() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    return {
        "/api/import": lambda body: engine.import_tree(body.get("data")),
        "/api/analyze": lambda body: engine.analyze_point(
            body.get("tree"), success_mode=body.get("success_mode", False),
            max_order=body.get("max_order", 6)),
        "/api/whatif": lambda body: engine.what_if(
            body.get("tree"), body.get("forced", {}),
            success_mode=body.get("success_mode", False)),
        "/api/uncertainty": lambda body: engine.analyze_uncertainty(
            body.get("tree"), distributions=body.get("distributions"),
            samples=body.get("samples"), n=body.get("n", engine.DEFAULT_SAMPLES),
            seed=body.get("seed", 0), success_mode=body.get("success_mode", False)),
        "/api/probfile": lambda body: engine.read_prob_upload(
            body.get("filename"), body.get("content_b64")),
    }


def info() -> Dict[str, Any]:
    return {"version": _faultree_version(),
            "examples": engine.examples_dir() is not None,
            "limits": {"max_samples": engine.MAX_SAMPLES,
                       "default_samples": engine.DEFAULT_SAMPLES,
                       "bdd_draw_limit": engine.BDD_DRAW_LIMIT}}


class Handler(BaseHTTPRequestHandler):
    server_version = "faultree-gui"
    protocol_version = "HTTP/1.1"
    routes = _post_routes()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass  # quiet; unexpected errors are printed with a traceback

    # -- responses ---------------------------------------------------------
    def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_bytes(code, body, "application/json; charset=utf-8")

    def send_error_json(self, code: int, message: str) -> None:
        self.send_json(code, {"error": message})

    def send_static(self, name: str) -> None:
        target = find_static(name)
        if target is None:
            self.send_error_json(404, "not found")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            ctype = "text/javascript"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json", "svg+xml")):
            ctype += "; charset=utf-8"
        self.send_bytes(200, target.read_bytes(), ctype)

    # -- checks ------------------------------------------------------------
    def host_allowed(self) -> bool:
        """Refuse DNS-rebinding requests when bound to the loopback interface:
        the Host header must then name the loopback too."""
        bound = self.server.server_address[0]
        if bound not in LOOPBACK:
            return True
        host = (self.headers.get("Host") or "").strip()
        if host.startswith("["):
            name = host[1:].split("]", 1)[0]
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name.lower() in LOOPBACK

    def read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise engine.GuiError(f"request too large (limit {MAX_BODY // 2**20} MB)")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise engine.GuiError(f"the request body is not valid JSON: {error}") from None
        if not isinstance(body, dict):
            raise engine.GuiError("the request body must be a JSON object")
        return body

    def run(self, action: Callable[[], Any]) -> None:
        try:
            self.send_json(200, action())
        except (ValueError, KeyError, TypeError, ImportError) as error:
            self.send_error_json(400, str(error) or repr(error))
        except RecursionError:
            self.send_error_json(400, "the tree is too deep for the recursive engine")
        except Exception:  # pragma: no cover - defensive
            traceback.print_exc()
            self.send_error_json(500, "internal error (see the terminal)")

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self.host_allowed():
            self.send_error_json(403, "forbidden host")
            return
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.send_static("index.html")
        elif path.startswith("/static/"):
            self.send_static(unquote(path[len("/static/"):]))
        elif path == "/api/info":
            self.run(info)
        elif path == "/api/examples":
            self.run(engine.list_examples)
        elif path.startswith("/api/examples/"):
            name = unquote(path[len("/api/examples/"):])
            self.run(lambda: engine.load_example(name))
        else:
            self.send_error_json(404, "not found")

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        if not self.host_allowed():
            self.send_error_json(403, "forbidden host")
            return
        route = self.routes.get(urlsplit(self.path).path)
        if route is None:
            self.send_error_json(404, "not found")
            return
        try:
            body = self.read_body()
        except (ValueError, TypeError) as error:
            self.send_error_json(400, str(error))
            return
        self.run(lambda: route(body))


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> GuiServer:
    """Bind the server without serving (tests use port 0)."""
    return GuiServer((host, port), Handler)


def _url(server: GuiServer) -> str:
    host, port = server.server_address[:2]
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/"


def serve(host: str = "127.0.0.1", port: Optional[int] = None,
          open_browser: bool = True) -> int:
    """Run the GUI until Ctrl+C. Without an explicit port, the default one is
    tried first and any free port is used if it is taken."""
    try:
        server = make_server(host, DEFAULT_PORT if port is None else port)
    except OSError as error:
        if port is not None or error.errno != errno.EADDRINUSE:
            print(f"Error: cannot listen on {host}:{port}: {error}", file=sys.stderr)
            return 1
        server = make_server(host, 0)
    url = _url(server)
    print(f"faultree GUI: {url}", flush=True)
    if host not in LOOPBACK:
        print("Warning: listening beyond this computer; anyone who reaches this"
              " address can use the GUI.", file=sys.stderr)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="faultree gui",
        description="Open the faultree graphical interface in the browser (local server).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: 127.0.0.1, this computer only)")
    parser.add_argument("--port", type=int, default=None,
                        help=f"port (default: {DEFAULT_PORT}, or a free one if it is taken)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    args = parser.parse_args(argv)
    return serve(args.host, args.port, open_browser=not args.no_browser)

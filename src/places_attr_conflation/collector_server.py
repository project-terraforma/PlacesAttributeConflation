"""Local web server for collecting replay evidence from conflict dork batches."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .collector import load_collector_cases, write_collector_payload
from .replay import dump_replay_corpus, load_replay_corpus


_HERE = Path(__file__).resolve().parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json_response(handler: BaseHTTPRequestHandler, payload: object, *, status: int = 200) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class CollectorHandler(BaseHTTPRequestHandler):
    server_version = "PlacesAttrCollector/1.0"

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content = path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (ctype or "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/cases":
            payload = getattr(self.server, "collector_payload", None)
            if payload is None:
                _json_response(self, {"error": "collector payload missing"}, status=500)
                return
            _json_response(self, payload)
            return
        if parsed.path == "/":
            self._send_file(getattr(self.server, "index_html_path"))
            return
        if parsed.path.startswith("/static/"):
            static_root = getattr(self.server, "static_root")
            self._send_file(static_root / parsed.path.removeprefix("/static/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, {"error": "invalid json"}, status=400)
            return

        out_dir: Path = getattr(self.server, "output_dir")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / payload.get("file_name", "replay_collected.json")
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")

        # Validate by attempting to load into the stable replay schema.
        tmp_path = out_path.with_suffix(".tmp.json")
        tmp_path.write_text(json.dumps(payload.get("replay_corpus", {}), indent=2, sort_keys=True), encoding="utf-8")
        try:
            episodes = load_replay_corpus(tmp_path)
            dump_replay_corpus(episodes, tmp_path)
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        _json_response(self, {"saved": str(out_path)})

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep console quiet; the harness prints a URL.
        return


def _write_static_index(root: Path) -> Path:
    html = _read_text(_HERE / "static" / "collector_index.html")
    out = root / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def run_collector_server(
    *,
    batch_csv: str | Path,
    output_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8844,
    export_payload: bool = True,
) -> ThreadingHTTPServer:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_collector_cases(batch_csv)
    payload_path = out_dir / "collector_cases.json"
    collector_payload = write_collector_payload(cases, payload_path) if export_payload else payload_path

    static_root = _HERE / "static"
    index_html_path = _write_static_index(out_dir)

    server = ThreadingHTTPServer((host, port), CollectorHandler)
    server.output_dir = out_dir
    server.static_root = static_root
    server.index_html_path = index_html_path
    server.collector_payload = json.loads(Path(collector_payload).read_text(encoding="utf-8"))
    return server


def serve_forever(server: ThreadingHTTPServer) -> None:
    server.serve_forever(poll_interval=0.2)


def start_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=serve_forever, args=(server,), daemon=True)
    thread.start()
    return thread


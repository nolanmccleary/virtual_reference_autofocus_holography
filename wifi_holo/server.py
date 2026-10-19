"""Local HTTP server exposing the compute engine to the browser app.

Stdlib only (no Flask). All physics runs here in Python; the page is a thin
client that POSTs parameters and GETs reconstruction slices.

Endpoints
---------
GET  /                 -> the live app page (viz3d/app.html)
GET  /api/state        -> current parameters + geometry + z axis
POST /api/params       -> {carrier_ghz, bandwidth_mhz, n_freq, nx,
                           z_emitter_cm, z_object_cm, cross_arm_cm}; rebuilds engine
GET  /api/recon?mode=&zi=&k=&hol=0|1
                       -> {slice, profile, z, zIndex[, amp, phase]}
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .engine import HoloEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_HTML = os.path.join(ROOT, "viz3d", "app.html")

DEFAULT_SCENE = "cross"
_engine = HoloEngine(DEFAULT_SCENE)
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def handle_one_request(self):
        # The browser routinely cancels in-flight requests (reload, or superseding
        # a slow recon); the resulting write to a closed socket raises ConnectionError.
        # That's benign -- swallow it instead of dumping a traceback per aborted request.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    # -------------------------------------------------------------- helpers
    def _write(self, body):
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True   # client went away mid-response

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except FileNotFoundError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write(body)

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            with _lock:
                _engine.set_scene(DEFAULT_SCENE)   # every page load resets to defaults
            self._send_file(APP_HTML, "text/html; charset=utf-8")
            return
        if u.path == "/api/state":
            with _lock:
                self._send_json(_engine.state())
            return
        if u.path == "/api/slice":
            q = parse_qs(u.query)
            mode = q.get("mode", ["single"])[0]
            zi = int(q.get("zi", ["0"])[0])
            k = int(q.get("k", ["0"])[0])
            with _lock:
                try:
                    out = _engine.slice_only(mode, zi, k)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, code=500)
                    return
            self._send_json(out)
            return
        if u.path == "/api/recon":
            q = parse_qs(u.query)
            mode = q.get("mode", ["single"])[0]
            zi = int(q.get("zi", ["0"])[0])
            k = int(q.get("k", ["0"])[0])
            want_hol = q.get("hol", ["0"])[0] == "1"
            with _lock:
                try:
                    out = _engine.recon(mode, zi, k)
                    if want_hol:
                        out.update(_engine.hologram_pngs(k))
                except Exception as exc:  # surface compute errors to the client
                    self._send_json({"error": str(exc)}, code=500)
                    return
            self._send_json(out)
            return
        self.send_error(404, "not found")

    # ------------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/api/params", "/api/scene"):
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "bad json"}, code=400)
            return
        with _lock:
            try:
                if u.path == "/api/scene":
                    # switch scene (optionally with initial params)
                    _engine.set_scene(str(body.get("name", "cross")), body.get("params"))
                else:
                    # merge changed params into the current scene (engine filters keys)
                    merged = dict(_engine.p)
                    merged.update(body or {})
                    _engine.set_params(merged)
            except Exception as exc:
                self._send_json({"error": str(exc)}, code=400)
                return
            self._send_json(_engine.state())


def serve(host: str = "127.0.0.1", port: int = 8777) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd

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

from .engine import HoloEngine, Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_HTML = os.path.join(ROOT, "viz3d", "app.html")

_engine = HoloEngine(Params())
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    # -------------------------------------------------------------- helpers
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        self.wfile.write(body)

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
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
        if u.path != "/api/params":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "bad json"}, code=400)
            return
        with _lock:
            cur = _engine.p
            g = body.get  # shorthand
            p = Params(
                carrier_ghz=float(g("carrier_ghz", cur.carrier_ghz)),
                bandwidth_mhz=float(g("bandwidth_mhz", cur.bandwidth_mhz)),
                n_freq=int(g("n_freq", cur.n_freq)),
                nx=int(g("nx", cur.nx)),
                z_emitter_cm=float(g("z_emitter_cm", cur.z_emitter_cm)),
                z_object_cm=float(g("z_object_cm", cur.z_object_cm)),
                cross_arm_cm=float(g("cross_arm_cm", cur.cross_arm_cm)),
                cross_thickness_cm=float(g("cross_thickness_cm", cur.cross_thickness_cm)),
                emitter_x_cm=float(g("emitter_x_cm", cur.emitter_x_cm)),
                emitter_y_cm=float(g("emitter_y_cm", cur.emitter_y_cm)),
                cross_x_cm=float(g("cross_x_cm", cur.cross_x_cm)),
                cross_y_cm=float(g("cross_y_cm", cur.cross_y_cm)),
                ref_x_cm=float(g("ref_x_cm", cur.ref_x_cm)),
                ref_y_cm=float(g("ref_y_cm", cur.ref_y_cm)),
            )
            _engine.set_params(p)
            self._send_json(_engine.state())


def serve(host: str = "127.0.0.1", port: int = 8777) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd

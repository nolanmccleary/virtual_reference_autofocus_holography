#!/usr/bin/env python3
"""Single entry point for the Wi-Fi holography reproduction.

    python3 run.py              # launch the live simulator (local server) + open browser
    python3 run.py --port 9000  # use a different port
    python3 run.py --no-open    # start the server but don't open a browser
    python3 run.py --static     # instead: build the self-contained precomputed artifact + open it

The live simulator runs all physics in Python; the browser is a thin client.
GPU acceleration (PyTorch/MPS) for the walk-scene backprojection kicks in
automatically when the project venv (.venv, with torch installed) is present.
"""

import os
import sys

# Re-exec under the project venv (which has torch/MPS) so `python3 run.py`
# transparently gets GPU acceleration; harmless if no venv. Compare sys.prefix
# (the venv dir) rather than the interpreter path -- the venv's python symlinks
# to the same base binary, so an executable-path check would wrongly skip this.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV = os.path.join(_ROOT, ".venv")
_VENV_PY = os.path.join(_VENV, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    os.execv(_VENV_PY, [_VENV_PY, *sys.argv])

import argparse
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_static(open_browser: bool) -> None:
    from wifi_holo.scene import CrossScene
    from wifi_holo.export import payload_json
    template = os.path.join(ROOT, "viz3d", "scene_template.html")
    out = os.path.join(ROOT, "viz3d", "scene.html")
    with open(template, encoding="utf-8") as fh:
        html = fh.read().replace("/*__RECON_DATA__*/ null", payload_json(CrossScene()))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(out, ROOT)} ({len(html)/1e6:.2f} MB)")
    if open_browser:
        webbrowser.open("file://" + out)


def run_live(port: int, open_browser: bool) -> None:
    from wifi_holo.server import serve
    httpd = serve(port=port)
    url = f"http://127.0.0.1:{port}/"

    # Warm up the GPU (torch/MPS) in the background so the first walk-scene
    # reconstruction isn't blocked by Metal kernel compilation (~6 s one-off).
    def _warmup():
        try:
            from wifi_holo.engine import HoloEngine, _TORCH_DEVICE
            if _TORCH_DEVICE in ("mps", "cuda"):
                HoloEngine("walk", {"num_samples": 80, "n_steps": 4}).recon("single", 10, 80)
                print(f"  (GPU warmed up: {_TORCH_DEVICE})")
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()

    print(f"Wi-Fi holography simulator running at {url}")
    print("  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="Wi-Fi holography reproduction.")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--static", action="store_true", help="build the precomputed standalone artifact instead")
    args = ap.parse_args()

    if args.static:
        run_static(not args.no_open)
    else:
        run_live(args.port, not args.no_open)


if __name__ == "__main__":
    main()

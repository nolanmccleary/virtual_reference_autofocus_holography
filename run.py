#!/usr/bin/env python3
"""Single entry point for the Wi-Fi holography reproduction.

    python3 run.py            # build the cross scene + open the interactive viewer
    python3 run.py --no-open  # build only, don't launch a browser
    python3 run.py --figures  # also regenerate the static PNG figures

Everything runs the Python simulation (the single source of truth) and writes
the self-contained artifact to viz3d/scene.html.
"""

import argparse
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from wifi_holo.scene import CrossScene        # noqa: E402
from wifi_holo.export import payload_json      # noqa: E402

TEMPLATE = os.path.join(ROOT, "viz3d", "scene_template.html")
ARTIFACT = os.path.join(ROOT, "viz3d", "scene.html")
PLACEHOLDER = "/*__RECON_DATA__*/ null"


def build_artifact(scene: CrossScene) -> str:
    print("simulating + serialising reconstruction ...")
    data = payload_json(scene)
    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in template")
    html = template.replace(PLACEHOLDER, data)
    with open(ARTIFACT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  wrote {os.path.relpath(ARTIFACT, ROOT)}  ({len(html)/1e6:.2f} MB)")
    return ARTIFACT


def build_figures(scene: CrossScene) -> None:
    from wifi_holo import viz
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    viz.plot_hologram(scene, scene.carrier_hz, os.path.join(ROOT, "figures", "cross_hologram.png"))
    viz.plot_reconstruction(scene, os.path.join(ROOT, "figures", "cross_reconstruction.png"))
    print("  wrote figures/cross_hologram.png, figures/cross_reconstruction.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + open the Wi-Fi holography cross scene.")
    ap.add_argument("--no-open", action="store_true", help="build only; do not open a browser")
    ap.add_argument("--figures", action="store_true", help="also regenerate static PNG figures")
    args = ap.parse_args()

    scene = CrossScene()
    path = build_artifact(scene)
    if args.figures:
        build_figures(scene)

    if not args.no_open:
        print(f"opening {os.path.relpath(path, ROOT)} ...")
        webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the self-contained interactive cross-scene artifact.

Runs the Python wave simulation (the single source of truth), serialises the
reconstruction into a JSON payload, injects it into the viewer template, and
writes viz3d/scene.html. Publish that file as the artifact.

Usage, from the repository root:
    python3 scripts/build_cross_artifact.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_holo.scene import CrossScene       # noqa: E402
from wifi_holo.export import payload_json     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(ROOT, "viz3d", "scene_template.html")
OUTPUT = os.path.join(ROOT, "viz3d", "scene.html")
PLACEHOLDER = "/*__RECON_DATA__*/ null"


def main() -> None:
    scene = CrossScene()
    print("simulating + serialising reconstruction ...")
    data = payload_json(scene)
    print(f"  payload: {len(data)/1e6:.2f} MB")

    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in template")

    html = template.replace(PLACEHOLDER, data)
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

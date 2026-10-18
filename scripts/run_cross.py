#!/usr/bin/env python3
"""Generate the cross-scene figures (hologram + reconstruction).

Usage, from the repository root:
    python3 scripts/run_cross.py
Writes PNGs into ./figures/.
"""

import os
import sys

# make the wifi_holo package importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_holo.scene import CrossScene          # noqa: E402
from wifi_holo import viz                        # noqa: E402


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    scene = CrossScene()

    viz.plot_hologram(scene, scene.carrier_hz, "figures/cross_hologram.png")
    print("wrote figures/cross_hologram.png")

    viz.plot_reconstruction(scene, "figures/cross_reconstruction.png", white_light=False)
    print("wrote figures/cross_reconstruction.png")


if __name__ == "__main__":
    main()

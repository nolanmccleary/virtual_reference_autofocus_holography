"""Export reconstruction data as a compact JSON payload for the web viewer.

Each reconstructed slice is rendered to a small colour-mapped PNG and embedded
as a base64 data URI. Python remains the single source of truth; the artifact
only displays what this produces.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .forward import record_hologram, reference_value  # noqa: E402
from .reconstruct import (  # noqa: E402
    dark_field_image,
    homodyne_normalize,
    reconstruct_depth_stack,
)
from .scene import CrossScene  # noqa: E402


def _png_b64(norm_img: np.ndarray, cmap: str) -> str:
    """norm_img already in [0,1]; return base64 PNG string (no data: prefix)."""
    buf = BytesIO()
    plt.imsave(buf, norm_img, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _slice_png(img: np.ndarray, cmap: str = "inferno", gamma: float = 0.7) -> str:
    """Per-slice normalisation + gamma so structure is visible at every depth."""
    m = img.max()
    a = (img / m) ** gamma if m > 0 else img
    return _png_b64(a, cmap)


def build_payload(scene: CrossScene, n_depths: int = 41) -> dict:
    z = np.linspace(0.6, 2.6, n_depths)

    vol_s = reconstruct_depth_stack(scene, z, white_light=False)
    vol_w = reconstruct_depth_stack(scene, z, white_light=True)

    def profile(vol):
        p = vol.reshape(len(z), -1).max(axis=1)
        return (p / p.max()).tolist()

    slices_s = [_slice_png(vol_s[i]) for i in range(len(z))]
    slices_w = [_slice_png(vol_w[i]) for i in range(len(z))]

    # dark-field at the object plane
    df = dark_field_image(scene, scene.z_object, white_light=True, mask_strength=0.9)
    df_png = _slice_png(df, cmap="inferno", gamma=0.6)
    i_obj = int(np.argmin(np.abs(z - scene.z_object)))

    # hologram amplitude + phase at carrier
    holo = homodyne_normalize(
        record_hologram(scene, scene.carrier_hz), reference_value(scene, scene.carrier_hz)
    )
    amp = np.abs(holo)
    amp_png = _png_b64(np.clip(amp / np.percentile(amp, 99), 0, 1), "viridis")
    phase_png = _png_b64((np.angle(holo) + np.pi) / (2 * np.pi), "twilight")

    return {
        "nx": scene.nx, "ny": scene.ny, "Lx": scene.Lx, "Ly": scene.Ly,
        "dx": scene.dx,
        "z": z.tolist(),
        "zObject": scene.z_object, "zEmitter": scene.z_emitter,
        "iObject": i_obj,
        "emitterXY": list(scene.emitter_xy),
        "refXY": list(scene.reference_xy),
        "crossCenter": list(scene.cross_center_xy),
        "armL": scene.cross_arm_length / 2.0, "armW": scene.cross_arm_width / 2.0,
        "carrierGHz": scene.carrier_hz / 1e9,
        "lambdaCm": scene.carrier_wavelength * 100.0,
        "bandwidthMHz": scene.bandwidth_hz / 1e6,
        "nFreq": scene.n_frequencies,
        "profile": {"single": profile(vol_s), "white": profile(vol_w)},
        "slices": {"single": slices_s, "white": slices_w},
        "darkfield": df_png,
        "hologram": {"amp": amp_png, "phase": phase_png},
    }


def payload_json(scene: CrossScene, n_depths: int = 41) -> str:
    return json.dumps(build_payload(scene, n_depths), separators=(",", ":"))

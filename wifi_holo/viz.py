"""Matplotlib figures: the recorded hologram and the reconstructions."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .forward import record_hologram, reference_value  # noqa: E402
from .reconstruct import (  # noqa: E402
    homodyne_normalize,
    reconstruct_depth_stack,
)
from .scene import CrossScene  # noqa: E402


def _extent(scene: CrossScene):
    return [0, scene.Lx, 0, scene.Ly]


def plot_hologram(scene: CrossScene, frequency_hz: float | None, path: str) -> None:
    """Amplitude and phase of the recorded hologram (the 'registered' wave front)."""
    f = frequency_hz if frequency_hz is not None else scene.carrier_hz
    holo = homodyne_normalize(record_hologram(scene, f), reference_value(scene, f))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = ax[0].imshow(np.abs(holo), extent=_extent(scene), origin="lower",
                       cmap="viridis", aspect="equal")
    ax[0].set_title("Recorded amplitude  |I|")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(np.angle(holo), extent=_extent(scene), origin="lower",
                       cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
    ax[1].set_title(r"Recorded phase  $\Phi$")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # mark emitter (projected) and reference antenna
    for a in ax:
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
        a.plot(*scene.emitter_xy, "r+", ms=12, mew=2, label="emitter (proj.)")
        a.plot(*scene.reference_xy, "wo", ms=7, mfc="none", mew=2, label="ref. antenna")
    ax[0].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Hologram on scan plane  (f = {f/1e9:.2f} GHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reconstruction(scene: CrossScene, path: str, white_light: bool = False) -> None:
    """Depth profile + emitter-plane focus + object-plane shadow."""
    z = np.linspace(0.6, 2.6, 41)
    vol = reconstruct_depth_stack(scene, z, white_light=white_light)

    # nearest indices for the two planes of interest
    i_obj = int(np.argmin(np.abs(z - scene.z_object)))
    i_emit = int(np.argmin(np.abs(z - scene.z_emitter)))

    peak = vol.reshape(len(z), -1).max(axis=1)
    peak = peak / peak.max()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].plot(z, peak, "-o", ms=3)
    ax[0].axvline(scene.z_emitter, color="r", ls="--", label="emitter 2.30 m")
    ax[0].axvline(scene.z_object, color="b", ls="--", label="object 1.40 m")
    ax[0].set_xlabel("reconstruction depth z (m)")
    ax[0].set_ylabel("peak intensity (norm.)")
    ax[0].set_title("Depth profile")
    ax[0].legend(fontsize=8)

    im1 = ax[1].imshow(vol[i_emit], extent=_extent(scene), origin="lower",
                       cmap="inferno", aspect="equal")
    ax[1].set_title(f"Emitter plane  z={z[i_emit]:.2f} m")
    ax[1].plot(*scene.emitter_xy, "c+", ms=12, mew=2)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(vol[i_obj], extent=_extent(scene), origin="lower",
                       cmap="inferno", aspect="equal")
    ax[2].set_title(f"Object plane  z={z[i_obj]:.2f} m  (shadow)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    for a in ax[1:]:
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")

    tag = "white-light" if white_light else "single frequency"
    fig.suptitle(f"Reconstruction by backpropagation  ({tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

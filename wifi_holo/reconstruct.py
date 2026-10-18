"""Digital reconstruction: homodyne normalisation, backpropagation,
white-light speckle suppression, and dark-field object enhancement.
"""

from __future__ import annotations

import numpy as np

from .forward import record_hologram, reference_value
from .propagation import angular_spectrum_propagate
from .scene import CrossScene
from .sources import wavelength_of


def homodyne_normalize(hologram: np.ndarray, ref_value: complex) -> np.ndarray:
    """Normalise the recorded field by the reference-antenna value (Eq. 1).

    Removes the unknown emitted signal U_em up to the irrelevant constant
    U(x_r, y_r, f)^-1. In simulation the emitted amplitude is already unity, so
    this is a global complex rescale; included for fidelity to the method.
    """
    if ref_value == 0:
        return hologram.copy()
    return hologram / ref_value


def backpropagate(
    hologram: np.ndarray, z: float, wavelength: float, dx: float, dy: float
) -> np.ndarray:
    """Reconstruct the field at a plane a distance ``z`` in front of the recorder
    (toward the source). Inverse of the forward propagation -> dz = -z."""
    return angular_spectrum_propagate(hologram, dz=-z, wavelength=wavelength, dx=dx, dy=dy)


def reconstruct_depth_stack(
    scene: CrossScene,
    z_values: np.ndarray,
    frequency_hz: float | None = None,
    white_light: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Return an intensity volume (len(z), ny, nx) reconstructed over depths.

    If ``white_light`` is True, incoherently sum intensities over the scene's
    frequency band at each depth (Sum_f |I(x,y,f)|^2), suppressing speckle.
    Otherwise use the single ``frequency_hz`` (defaults to the carrier).
    """
    if white_light:
        freqs = scene.frequencies()
    else:
        freqs = [frequency_hz if frequency_hz is not None else scene.carrier_hz]

    vol = np.zeros((len(z_values), scene.ny, scene.nx), dtype=float)
    for f in freqs:
        wl = wavelength_of(f)
        holo = record_hologram(scene, f)
        if normalize:
            holo = homodyne_normalize(holo, reference_value(scene, f))
        for i, z in enumerate(z_values):
            fieldz = backpropagate(holo, z, wl, scene.dx, scene.dy)
            vol[i] += np.abs(fieldz) ** 2
    return vol


def dark_field_image(
    scene: CrossScene,
    z: float,
    white_light: bool = True,
    mask_sigma: float | None = None,
    mask_strength: float = 1.0,
) -> np.ndarray:
    """Dark-field reconstruction at depth ``z`` (Eq. 3).

    Masks the emitter in its plane, forward-propagates the remaining field to
    the object plane, and subtracts its intensity from the direct reconstruction:
        |I_dark|^2 = |I_masked|^2 - |I_unmasked|^2    (clipped at 0)

    ``mask_sigma`` is the Gaussian radius (m) used to blank the emitter focus;
    ``mask_strength`` in (0, 1] blocks only part of the direct light to keep a
    wide dark-field cone.
    """
    if mask_sigma is None:
        mask_sigma = 6.0 * scene.carrier_wavelength  # a few wi-fi wavelengths

    freqs = scene.frequencies() if white_light else [scene.carrier_hz]

    direct = np.zeros((scene.ny, scene.nx), dtype=float)
    masked = np.zeros((scene.ny, scene.nx), dtype=float)

    X, Y = scene.grids()
    ex, ey = scene.emitter_xy
    gauss = np.exp(-((X - ex) ** 2 + (Y - ey) ** 2) / (2.0 * mask_sigma ** 2))
    emitter_mask = 1.0 - mask_strength * gauss  # ~0 at emitter, 1 far away

    for f in freqs:
        wl = wavelength_of(f)
        holo = record_hologram(scene, f)
        holo = homodyne_normalize(holo, reference_value(scene, f))

        # direct reconstruction at the object plane
        u_obj_direct = backpropagate(holo, z, wl, scene.dx, scene.dy)
        direct += np.abs(u_obj_direct) ** 2

        # reconstruct to emitter plane, blank the source, forward-propagate back
        u_emit = backpropagate(holo, scene.z_emitter, wl, scene.dx, scene.dy)
        u_emit_masked = u_emit * emitter_mask
        # emitter plane -> object plane: travels distance (z_emitter - z)
        u_obj_masked = angular_spectrum_propagate(
            u_emit_masked, dz=(scene.z_emitter - z), wavelength=wl,
            dx=scene.dx, dy=scene.dy,
        )
        masked += np.abs(u_obj_masked) ** 2

    return np.clip(masked - direct, 0.0, None)

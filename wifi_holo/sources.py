"""Coherent sources and frequency helpers."""

from __future__ import annotations

import numpy as np

speed_of_light = 299_792_458.0  # m/s


def wavelength_of(frequency_hz: float) -> float:
    """Wavelength (m) for a frequency in Hz."""
    return speed_of_light / frequency_hz


def point_source_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_plane: float,
    emitter_xyz: tuple[float, float, float],
    wavelength: float,
) -> np.ndarray:
    """Complex field of a diverging point source, sampled on a plane at ``z_plane``.

    Returns exp(+i k R) / R where R is the distance from the emitter to each
    grid point, with k = 2*pi/wavelength. This is the exact free-space Green's
    function (scalar), used to launch the wave analytically from the emitter to
    the first plane before angular-spectrum propagation takes over.
    """
    ex, ey, ez = emitter_xyz
    R = np.sqrt((grid_x - ex) ** 2 + (grid_y - ey) ** 2 + (z_plane - ez) ** 2)
    k = 2.0 * np.pi / wavelength
    return np.exp(1j * k * R) / R

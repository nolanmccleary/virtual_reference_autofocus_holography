"""Scalar wave propagation backends.

Currently the angular-spectrum method (Fast Fourier Transform based). The
signature is kept deliberately generic so an alternative backend -- e.g. a
Finite-Difference Time-Domain (FDTD) solver -- can be swapped in later behind
the same `propagate(field, dz, ...)` contract.

Convention
----------
Fields are 2D complex arrays sampled on a regular grid, indexed [y, x].
Time convention is exp(-i*omega*t); a diverging spherical wave is
exp(+i*k*r)/r. `dz > 0` propagates in the direction of travel (away from the
source); `dz < 0` back-propagates toward the source. Evanescent components
always decay (never blow up), matching Eq. (2) of the paper.
"""

from __future__ import annotations

import numpy as np


def angular_spectrum_propagate(
    field: np.ndarray,
    dz: float,
    wavelength: float,
    dx: float,
    dy: float | None = None,
    pad: bool = True,
) -> np.ndarray:
    """Propagate a scalar complex field a distance ``dz`` by the angular spectrum.

    Parameters
    ----------
    field : (Ny, Nx) complex ndarray
        Complex field in the source plane, indexed [y, x].
    dz : float
        Propagation distance (metres). Positive = forward (away from source),
        negative = backward (toward source / reconstruction).
    wavelength : float
        Wavelength (metres).
    dx, dy : float
        Sample spacing (metres). ``dy`` defaults to ``dx``.
    pad : bool
        Zero-pad to twice the size before the transform to suppress circular
        wrap-around, then crop back. Recommended.
    """
    if dy is None:
        dy = dx

    ny, nx = field.shape
    if pad:
        py, px = ny // 2, nx // 2
    else:
        py = px = 0

    fp = np.pad(field, ((py, py), (px, px)))
    Ny, Nx = fp.shape

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 1.0 / wavelength  # spatial frequency magnitude of a plane wave (1/m)
    arg = k * k - FX * FX - FY * FY  # (kz / 2pi)^2 ; >=0 propagating, <0 evanescent

    kz = np.sqrt(np.maximum(arg, 0.0))          # real part -> propagating phase
    decay = np.sqrt(np.maximum(-arg, 0.0))      # evanescent decay rate

    H = np.where(
        arg >= 0.0,
        np.exp(1j * 2.0 * np.pi * dz * kz),
        np.exp(-2.0 * np.pi * abs(dz) * decay),  # abs(dz): always decays
    )

    spectrum = np.fft.fft2(fp)
    out = np.fft.ifft2(spectrum * H)

    if pad:
        out = out[py:py + ny, px:px + nx]
    return out

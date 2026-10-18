"""Scene / geometry definitions.

All lengths in metres. The recording plane sits at z = 0. Positive z points
from the recorder toward the emitter (into the scene), so:

    recorder plane   z = 0
    object plane     z = z_object   (e.g. 1.40 m)
    emitter          z = z_emitter  (e.g. 2.30 m)

This matches the cross scene of PRL 118, 183901: emitter 230 cm from the
recording plane, object 90 cm in front of the emitter (-> 140 cm from recorder).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sources import wavelength_of


@dataclass
class CrossScene:
    # --- recording plane (synthetic-aperture scan region) ---
    Lx: float = 3.0          # width  (m)  [paper: 3 m]
    Ly: float = 2.0          # height (m)  [paper: 2 m]
    dx: float = 0.025        # sample spacing (m); < lambda/2 at 5 GHz (3 cm)

    # --- depths ---
    z_object: float = 1.40   # object (cross) plane (m)
    z_emitter: float = 2.30  # emitter distance from recorder (m) [paper: 230 cm]

    # --- emitter lateral position (defaults to plane centre) ---
    emitter_xy: tuple[float, float] | None = None

    # --- cross object geometry ---
    cross_center_xy: tuple[float, float] | None = None
    cross_arm_length: float = 1.00   # full span of each arm (m)
    cross_arm_width: float = 0.20    # arm thickness (m)

    # --- frequency / illumination ---
    carrier_hz: float = 5.0e9        # 5 GHz Wi-Fi channel
    bandwidth_hz: float = 70.0e6     # white-light band (paper: ~70 MHz)
    n_frequencies: int = 32          # samples across the band

    # --- reference antenna (homodyne denominator) ---
    reference_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        if self.emitter_xy is None:
            self.emitter_xy = (cx, cy)
        if self.cross_center_xy is None:
            self.cross_center_xy = (cx, cy)
        if self.reference_xy is None:
            # off to one side, near an edge of the recording plane
            self.reference_xy = (0.15 * self.Lx, 0.15 * self.Ly)

    # ------------------------------------------------------------------ grids
    @property
    def nx(self) -> int:
        return int(round(self.Lx / self.dx))

    @property
    def ny(self) -> int:
        return int(round(self.Ly / self.dx))

    @property
    def dy(self) -> float:
        return self.dx

    def grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) meshgrids of plane coordinates, shape (ny, nx)."""
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dx
        return np.meshgrid(x, y)

    # ---------------------------------------------------------------- objects
    def cross_mask(self) -> np.ndarray:
        """Amplitude transmission of the cross object plane: 0 = opaque metal,
        1 = free space. Shape (ny, nx)."""
        X, Y = self.grids()
        ccx, ccy = self.cross_center_xy
        half_len = self.cross_arm_length / 2.0
        half_w = self.cross_arm_width / 2.0

        horiz = (np.abs(X - ccx) <= half_len) & (np.abs(Y - ccy) <= half_w)
        vert = (np.abs(Y - ccy) <= half_len) & (np.abs(X - ccx) <= half_w)
        opaque = horiz | vert

        t = np.ones((self.ny, self.nx), dtype=float)
        t[opaque] = 0.0
        return t

    # ------------------------------------------------------------ frequencies
    def frequencies(self) -> np.ndarray:
        """Frequency samples spanning the white-light band about the carrier."""
        if self.n_frequencies <= 1:
            return np.array([self.carrier_hz])
        return np.linspace(
            self.carrier_hz - self.bandwidth_hz / 2.0,
            self.carrier_hz + self.bandwidth_hz / 2.0,
            self.n_frequencies,
        )

    @property
    def emitter_xyz(self) -> tuple[float, float, float]:
        ex, ey = self.emitter_xy
        return (ex, ey, self.z_emitter)

    @property
    def carrier_wavelength(self) -> float:
        return wavelength_of(self.carrier_hz)

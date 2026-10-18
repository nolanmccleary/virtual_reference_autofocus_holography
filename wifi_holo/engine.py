"""Interactive compute engine with partial-aperture (progressive-scan) support.

The engine caches the full forward complex hologram per frequency for the
current parameters. Reconstruction is then done from a *partial* aperture --
the first ``k`` scan points in raster order -- which is what makes the scan
build up the image in real time: more acquired points -> sharper reconstruction.

Python stays the single source of truth; the HTTP server is a thin wrapper.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .forward import record_hologram, reference_value  # noqa: E402
from .propagation import angular_spectrum_propagate  # noqa: E402
from .scene import CrossScene  # noqa: E402
from .sources import speed_of_light, wavelength_of  # noqa: E402

LX, LY = 3.0, 2.0  # recording plane size is fixed (m)


@dataclass
class Params:
    carrier_ghz: float = 5.0
    bandwidth_mhz: float = 70.0
    n_freq: int = 32
    nx: int = 120              # scan-plane columns (resolution); ny derived
    z_emitter_cm: float = 230.0
    z_object_cm: float = 140.0
    cross_arm_cm: float = 100.0   # full arm span
    cross_thickness_cm: float = 5.0
    # lateral positions (cm) on / relative to the recording plane
    emitter_x_cm: float = 150.0
    emitter_y_cm: float = 100.0
    cross_x_cm: float = 150.0
    cross_y_cm: float = 100.0
    ref_x_cm: float = 45.0
    ref_y_cm: float = 30.0
    n_depths: int = 41

    def scene(self) -> CrossScene:
        dx = LX / self.nx
        return CrossScene(
            Lx=LX, Ly=LY, dx=dx,
            z_object=self.z_object_cm / 100.0,
            z_emitter=self.z_emitter_cm / 100.0,
            carrier_hz=self.carrier_ghz * 1e9,
            bandwidth_hz=self.bandwidth_mhz * 1e6,
            n_frequencies=int(self.n_freq),
            cross_arm_length=self.cross_arm_cm / 100.0,
            cross_thickness=self.cross_thickness_cm / 100.0,
            emitter_xy=(self.emitter_x_cm / 100.0, self.emitter_y_cm / 100.0),
            cross_center_xy=(self.cross_x_cm / 100.0, self.cross_y_cm / 100.0),
            reference_xy=(self.ref_x_cm / 100.0, self.ref_y_cm / 100.0),
        )


def _png_b64(norm01: np.ndarray, cmap: str) -> str:
    buf = BytesIO()
    plt.imsave(buf, norm01, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class HoloEngine:
    def __init__(self, params: Params | None = None):
        self.set_params(params or Params())

    # ---------------------------------------------------------------- params
    def set_params(self, params: Params) -> None:
        self.p = params
        self.scene = params.scene()
        s = self.scene
        self.nx, self.ny = s.nx, s.ny
        self.total = self.nx * self.ny
        self.freqs = s.frequencies()
        self.carrier_idx = int(np.argmin(np.abs(self.freqs - s.carrier_hz)))
        self.z_values = np.linspace(0.4, s.z_emitter + 0.3, params.n_depths)
        self.i_object = int(np.argmin(np.abs(self.z_values - s.z_object)))
        self.i_emitter = int(np.argmin(np.abs(self.z_values - s.z_emitter)))

        # cache the full complex hologram + reference for every frequency
        self.U = [record_hologram(s, f) for f in self.freqs]
        self.ref = [reference_value(s, f) or 1.0 for f in self.freqs]

        # emitter-plane Gaussian mask for dark-field
        X, Y = s.grids()
        ex, ey = s.emitter_xy
        sigma = 6.0 * s.carrier_wavelength
        self.emitter_mask = 1.0 - 0.9 * np.exp(-((X - ex) ** 2 + (Y - ey) ** 2) / (2 * sigma ** 2))

    # -------------------------------------------------------------- aperture
    def _aperture(self, k: int) -> np.ndarray:
        k = int(np.clip(k, 0, self.total))
        return (np.arange(self.total) < k).reshape(self.ny, self.nx).astype(float)

    def _bp(self, holo: np.ndarray, z: float, wl: float) -> np.ndarray:
        return angular_spectrum_propagate(holo, dz=-z, wavelength=wl, dx=self.scene.dx, dy=self.scene.dy)

    # --------------------------------------------------------------- stacks
    def _stack(self, mode: str, k: int) -> np.ndarray:
        s = self.scene
        mask = self._aperture(k)
        nz = len(self.z_values)
        vol = np.zeros((nz, self.ny, self.nx))

        if mode == "single":
            holo = (self.U[self.carrier_idx] * mask) / self.ref[self.carrier_idx]
            wl = wavelength_of(self.freqs[self.carrier_idx])
            for i, z in enumerate(self.z_values):
                vol[i] = np.abs(self._bp(holo, z, wl)) ** 2

        elif mode == "white":
            for fi, f in enumerate(self.freqs):
                holo = (self.U[fi] * mask) / self.ref[fi]
                wl = wavelength_of(f)
                for i, z in enumerate(self.z_values):
                    vol[i] += np.abs(self._bp(holo, z, wl)) ** 2

        elif mode == "dark":
            masked_sum = np.zeros_like(vol)
            direct_sum = np.zeros_like(vol)
            for fi, f in enumerate(self.freqs):
                holo = (self.U[fi] * mask) / self.ref[fi]
                wl = wavelength_of(f)
                u_emit = self._bp(holo, s.z_emitter, wl) * self.emitter_mask
                for i, z in enumerate(self.z_values):
                    direct_sum[i] += np.abs(self._bp(holo, z, wl)) ** 2
                    u_obj = angular_spectrum_propagate(
                        u_emit, dz=(s.z_emitter - z), wavelength=wl, dx=s.dx, dy=s.dy)
                    masked_sum[i] += np.abs(u_obj) ** 2
            vol = np.clip(masked_sum - direct_sum, 0.0, None)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return vol

    def _recon_one(self, mode: str, z: float, k: int) -> np.ndarray:
        """Intensity at a SINGLE depth (used for fast z-scrubbing)."""
        s = self.scene
        mask = self._aperture(k)
        if mode == "single":
            wl = wavelength_of(self.freqs[self.carrier_idx])
            holo = (self.U[self.carrier_idx] * mask) / self.ref[self.carrier_idx]
            return np.abs(self._bp(holo, z, wl)) ** 2
        if mode == "white":
            acc = np.zeros((self.ny, self.nx))
            for fi, f in enumerate(self.freqs):
                holo = (self.U[fi] * mask) / self.ref[fi]
                acc += np.abs(self._bp(holo, z, wavelength_of(f))) ** 2
            return acc
        if mode == "dark":
            masked = np.zeros((self.ny, self.nx))
            direct = np.zeros((self.ny, self.nx))
            for fi, f in enumerate(self.freqs):
                wl = wavelength_of(f)
                holo = (self.U[fi] * mask) / self.ref[fi]
                u_emit = self._bp(holo, s.z_emitter, wl) * self.emitter_mask
                direct += np.abs(self._bp(holo, z, wl)) ** 2
                masked += np.abs(angular_spectrum_propagate(
                    u_emit, dz=(s.z_emitter - z), wavelength=wl, dx=s.dx, dy=s.dy)) ** 2
            return np.clip(masked - direct, 0.0, None)
        raise ValueError(f"unknown mode {mode!r}")

    def slice_only(self, mode: str, z_index: int, k: int, gamma: float = 0.7) -> dict:
        """Fast single-depth reconstruction (no profile) for z-scrubbing."""
        z_index = int(np.clip(z_index, 0, len(self.z_values) - 1))
        z = float(self.z_values[z_index])
        img = self._recon_one(mode, z, k)
        sl = (img / (img.max() or 1.0)) ** gamma
        return {"slice": _png_b64(sl, "inferno"), "z": z, "zIndex": z_index}

    # ---------------------------------------------------------------- public
    def recon(self, mode: str, z_index: int, k: int, gamma: float = 0.7) -> dict:
        vol = self._stack(mode, k)
        gmax = vol.max() or 1.0
        # depth-focus curve uses the global max; the displayed slice is
        # normalised to ITS OWN max so every depth is vivid (as in the paper).
        prof = (vol.reshape(len(self.z_values), -1).max(axis=1) / gmax)
        z_index = int(np.clip(z_index, 0, len(self.z_values) - 1))
        sl = vol[z_index]
        slice01 = (sl / (sl.max() or 1.0)) ** gamma
        return {
            "slice": _png_b64(slice01, "inferno"),
            "profile": prof.tolist(),
            "zIndex": z_index,
            "z": float(self.z_values[z_index]),
        }

    def hologram_pngs(self, k: int) -> dict:
        """Masked amplitude + phase at the carrier (the wave front being recorded)."""
        mask = self._aperture(k)
        holo = (self.U[self.carrier_idx] * mask) / self.ref[self.carrier_idx]
        amp = np.abs(holo)
        amp01 = np.clip(amp / (np.percentile(amp[mask > 0], 99) if mask.any() else 1.0), 0, 1)
        ph = (np.angle(holo) + np.pi) / (2 * np.pi) * mask  # unacquired -> 0
        return {"amp": _png_b64(amp01, "viridis"), "phase": _png_b64(ph, "twilight")}

    def state(self) -> dict:
        s = self.scene
        return {
            "nx": self.nx, "ny": self.ny, "total": self.total,
            "Lx": LX, "Ly": LY, "dx": s.dx,
            "z": self.z_values.tolist(),
            "iObject": self.i_object, "iEmitter": self.i_emitter,
            "zObject": s.z_object, "zEmitter": s.z_emitter,
            "emitterXY": list(s.emitter_xy), "refXY": list(s.reference_xy),
            "crossCenter": list(s.cross_center_xy),
            "armL": s.cross_arm_length / 2.0, "armW": s.cross_arm_width / 2.0,
            "crossThickness": s.cross_thickness,
            "params": {
                "carrier_ghz": self.p.carrier_ghz, "bandwidth_mhz": self.p.bandwidth_mhz,
                "n_freq": self.p.n_freq, "nx": self.p.nx,
                "z_emitter_cm": self.p.z_emitter_cm, "z_object_cm": self.p.z_object_cm,
                "cross_arm_cm": self.p.cross_arm_cm, "cross_thickness_cm": self.p.cross_thickness_cm,
                "emitter_x_cm": self.p.emitter_x_cm, "emitter_y_cm": self.p.emitter_y_cm,
                "cross_x_cm": self.p.cross_x_cm, "cross_y_cm": self.p.cross_y_cm,
                "ref_x_cm": self.p.ref_x_cm, "ref_y_cm": self.p.ref_y_cm,
            },
            "lambdaCm": wavelength_of(s.carrier_hz) * 100.0,
            "cLight": speed_of_light,
        }

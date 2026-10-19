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

from .propagation import angular_spectrum_propagate  # noqa: E402
from .sources import speed_of_light, wavelength_of  # noqa: E402
from . import scenes  # noqa: E402


def _png_b64(norm01: np.ndarray, cmap: str) -> str:
    buf = BytesIO()
    plt.imsave(buf, norm01, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class HoloEngine:
    """Scene-agnostic reconstruction engine.

    A scene module (wifi_holo.scenes.*) provides the forward physics -- the
    complex field per frequency at the recording plane plus geometry metadata.
    This class owns the shared synthetic-aperture + reconstruction machinery.
    """

    def __init__(self, scene_name: str = "cross", params: dict | None = None):
        self.set_scene(scene_name, params)

    # ---------------------------------------------------------------- params
    def set_scene(self, scene_name: str, params: dict | None = None) -> None:
        self.scene_name = scene_name
        self.builder = scenes.get(scene_name)
        p = dict(self.builder.defaults())
        if params:
            p.update({k: v for k, v in params.items() if k in p})
        self.set_params(p)

    def set_params(self, params: dict) -> None:
        # keep only keys this scene knows; fill the rest from defaults
        p = dict(self.builder.defaults())
        p.update({k: v for k, v in (params or {}).items() if k in p})
        self.p = p

        b = self.builder.build(p)
        self.built = b
        self.nx, self.ny = b.nx, b.ny
        self.dx, self.dy = b.dx, b.dy
        self.Lx, self.Ly = b.Lx, b.Ly
        self.freqs = b.freqs
        self.carrier_idx = b.carrier_idx
        self.z_values = b.z_values
        self.z_emitter = b.z_emitter

        # ---- scan grid (measurement points) at the requested separation ----
        sep_cm = float(p.get("point_sep_cm", b.dx * 100.0))
        sep = max(sep_cm / 100.0, b.dx)                   # not finer than the compute grid
        self.nx_pts = max(1, int(round(b.Lx / sep)))
        self.ny_pts = max(1, int(round(b.Ly / sep)))
        self.scan_total = self.nx_pts * self.ny_pts
        self.total = self.scan_total                       # progressive-scan count
        self.sep_x_cm = b.Lx / self.nx_pts * 100.0
        self.sep_y_cm = b.Ly / self.ny_pts * 100.0

        # map fine pixels <-> scan points (nearest), for block-fill sampling
        dx, dy = b.dx, b.dy
        xs = (np.arange(self.nx_pts) + 0.5) * b.Lx / self.nx_pts
        ys = (np.arange(self.ny_pts) + 0.5) * b.Ly / self.ny_pts
        scanpix_col = np.clip(np.round(xs / dx - 0.5).astype(int), 0, self.nx - 1)
        scanpix_row = np.clip(np.round(ys / dy - 0.5).astype(int), 0, self.ny - 1)
        fine_x = (np.arange(self.nx) + 0.5) * dx
        fine_y = (np.arange(self.ny) + 0.5) * dy
        owner_col = np.clip(np.round(fine_x / (b.Lx / self.nx_pts) - 0.5).astype(int), 0, self.nx_pts - 1)
        owner_row = np.clip(np.round(fine_y / (b.Ly / self.ny_pts) - 0.5).astype(int), 0, self.ny_pts - 1)
        self._srow = scanpix_row[owner_row]
        self._scol = scanpix_col[owner_col]
        self.owner_index = owner_row[:, None] * self.nx_pts + owner_col[None, :]

        # reference (homodyne denominator) + block-filled measured field per frequency
        rx = int(np.clip(round(b.reference_xy[0] / dx - 0.5), 0, self.nx - 1))
        ry = int(np.clip(round(b.reference_xy[1] / dy - 0.5), 0, self.ny - 1))
        self.ref = [complex(u[ry, rx]) or 1.0 for u in b.full_fields]
        self.U = [u[np.ix_(self._srow, self._scol)] for u in b.full_fields]

        # emitter-plane window that KEEPS just the focus, for dark-field illumination
        x = (np.arange(self.nx) + 0.5) * dx
        y = (np.arange(self.ny) + 0.5) * dy
        X, Y = np.meshgrid(x, y)
        ex, ey = b.emitter_xy
        sigma = b.carrier_wavelength
        self.emitter_pass = np.exp(-((X - ex) ** 2 + (Y - ey) ** 2) / (2 * sigma ** 2))

    # -------------------------------------------------------------- aperture
    def _aperture(self, k: int) -> np.ndarray:
        """Fine-grid mask of scan points acquired so far (first k in raster order)."""
        k = int(np.clip(k, 0, self.scan_total))
        return (self.owner_index < k).astype(float)

    def _bp(self, holo: np.ndarray, z: float, wl: float) -> np.ndarray:
        return angular_spectrum_propagate(holo, dz=-z, wavelength=wl, dx=self.dx, dy=self.dy)

    # --------------------------------------------------------------- stacks
    def _stack(self, mode: str, k: int) -> np.ndarray:
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
            # dark-field = illumination estimate - measured field (the light the
            # object removed -> object appears bright). Illumination is the emitter
            # focus alone, forward-propagated to each depth.
            illum_sum = np.zeros_like(vol)
            direct_sum = np.zeros_like(vol)
            for fi, f in enumerate(self.freqs):
                holo = (self.U[fi] * mask) / self.ref[fi]
                wl = wavelength_of(f)
                emitter_only = self._bp(holo, self.z_emitter, wl) * self.emitter_pass
                for i, z in enumerate(self.z_values):
                    illum = angular_spectrum_propagate(
                        emitter_only, dz=(self.z_emitter - z), wavelength=wl, dx=self.dx, dy=self.dy)
                    illum_sum[i] += np.abs(illum) ** 2
                    direct_sum[i] += np.abs(self._bp(holo, z, wl)) ** 2
            vol = np.clip(illum_sum - direct_sum, 0.0, None)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return vol

    def _recon_one(self, mode: str, z: float, k: int) -> np.ndarray:
        """Intensity at a SINGLE depth (used for fast z-scrubbing)."""
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
            illum = np.zeros((self.ny, self.nx))
            direct = np.zeros((self.ny, self.nx))
            for fi, f in enumerate(self.freqs):
                wl = wavelength_of(f)
                holo = (self.U[fi] * mask) / self.ref[fi]
                emitter_only = self._bp(holo, self.z_emitter, wl) * self.emitter_pass
                direct += np.abs(self._bp(holo, z, wl)) ** 2
                illum += np.abs(angular_spectrum_propagate(
                    emitter_only, dz=(self.z_emitter - z), wavelength=wl, dx=self.dx, dy=self.dy)) ** 2
            return np.clip(illum - direct, 0.0, None)
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
        # autofocus sharpness: summed squared intensity per slice, normalised by
        # the slice's total energy -> ΣI² / (ΣI)² (concentration, brightness-independent)
        flat = vol.reshape(len(self.z_values), -1)
        sharp = (flat ** 2).sum(axis=1) / np.maximum(flat.sum(axis=1), 1e-12) ** 2
        sharp_index = int(np.argmax(sharp))
        # lateral position of the focus in the sharpest slice (intensity centroid
        # of the bright peak -> sub-pixel x,y of the router)
        sharp_x, sharp_y = self._focus_xy(vol[sharp_index])
        z_index = int(np.clip(z_index, 0, len(self.z_values) - 1))
        sl = vol[z_index]
        slice01 = (sl / (sl.max() or 1.0)) ** gamma
        return {
            "slice": _png_b64(slice01, "inferno"),
            "profile": prof.tolist(),
            "sharpness": (sharp / (sharp.max() or 1.0)).tolist(),
            "sharpIndex": sharp_index,
            "sharpZ": float(self.z_values[sharp_index]),
            "sharpX": sharp_x,
            "sharpY": sharp_y,
            "zIndex": z_index,
            "z": float(self.z_values[z_index]),
        }

    def _focus_xy(self, img: np.ndarray) -> tuple[float, float]:
        """Lateral (x, y) in metres of the focal peak: intensity centroid within a
        window around the brightest pixel of the slice."""
        iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
        r = max(2, int(round(0.10 / self.dx)))     # ~10 cm window
        y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
        patch = img[y0:y1, x0:x1]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        w = patch.sum() or 1.0
        cx = float((xs * patch).sum() / w + 0.5) * self.dx
        cy = float((ys * patch).sum() / w + 0.5) * self.dy
        return cx, cy

    def hologram_pngs(self, k: int) -> dict:
        """Masked amplitude + phase at the carrier (the wave front being recorded)."""
        mask = self._aperture(k)
        holo = (self.U[self.carrier_idx] * mask) / self.ref[self.carrier_idx]
        amp = np.abs(holo)
        amp01 = np.clip(amp / (np.percentile(amp[mask > 0], 99) if mask.any() else 1.0), 0, 1)
        ph = (np.angle(holo) + np.pi) / (2 * np.pi) * mask  # unacquired -> 0
        return {"amp": _png_b64(amp01, "viridis"), "phase": _png_b64(ph, "twilight")}

    def state(self) -> dict:
        b = self.built
        st = {
            "scene": self.scene_name,
            "scenes": scenes.listing(),
            "schema": self.builder.PARAM_SCHEMA,
            "params": dict(self.p),
            "nx": self.nx, "ny": self.ny, "total": self.total,
            "Lx": b.Lx, "Ly": b.Ly, "dx": b.dx,
            "nxPts": self.nx_pts, "nyPts": self.ny_pts, "scanTotal": self.scan_total,
            "sepXcm": self.sep_x_cm, "sepYcm": self.sep_y_cm,
            "z": self.z_values.tolist(),
            "zEmitter": b.z_emitter,
            "emitterXY": list(b.emitter_xy), "refXY": list(b.reference_xy),
            "geometry": b.geometry,
            "lambdaCm": b.carrier_wavelength * 100.0,
            "cLight": speed_of_light,
        }
        st.update(b.scene_extra)
        return st

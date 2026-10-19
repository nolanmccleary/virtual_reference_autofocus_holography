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

# Optional PyTorch backend for GPU-accelerated backprojection (ALGORITHMS.md §7).
# Prefers Apple MPS, then CUDA, then torch-CPU; falls back to numpy if absent.
try:
    import os as _os
    import torch as _torch
    _override = _os.environ.get("PARALLAX_DEVICE", "").strip().lower()  # mps|cuda|cpu|numpy
    if _override == "numpy":
        _torch = None; _TORCH_DEVICE = None
    elif _override in ("mps", "cuda", "cpu"):
        _TORCH_DEVICE = _override
    elif _torch.backends.mps.is_available():
        _TORCH_DEVICE = "mps"
    elif _torch.cuda.is_available():
        _TORCH_DEVICE = "cuda"
    else:
        _TORCH_DEVICE = "cpu"
except Exception:  # torch not installed
    _torch = None
    _TORCH_DEVICE = None


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
        self._gen = getattr(self, "_gen", 0) + 1   # bump -> invalidates the volume cache
        self._vol_cache = {}                        # per (mode, k) volume bins

        b = self.builder.build(p)
        self.built = b
        self.nx, self.ny = b.nx, b.ny
        self.dx, self.dy = b.dx, b.dy
        self.Lx, self.Ly = b.Lx, b.Ly
        self.freqs = b.freqs
        self.carrier_idx = b.carrier_idx
        self.z_values = b.z_values
        self.z_emitter = b.z_emitter
        self.backend = b.backend

        if self.backend == "backprojection":
            self._setup_backprojection(b)
            return

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

    # --------------------------------------------------- backprojection setup
    def _setup_backprojection(self, b) -> None:
        self.sample_pos = b.sample_pos                 # (N, 3)
        self.sample_val = b.sample_val                 # per-freq (N,) complex
        self.scan_total = len(b.sample_pos)
        self.total = self.scan_total
        self.nx_pts, self.ny_pts = self.scan_total, 1  # readout as "N steps"
        self.sep_x_cm = self.sep_y_cm = 0.0
        X, Y = np.meshgrid(b.recon_x, b.recon_y)
        self._vx, self._vy = X, Y                      # voxel grids (ny, nx)
        # emitter position + per-sample distance, for dark-field (subtract direct)
        self._emit = np.array([b.emitter_xy[0], b.emitter_xy[1], b.z_emitter])
        self._samp_R = np.linalg.norm(self.sample_pos - self._emit, axis=1)

        # ---- GPU tensors (torch/MPS) for the backprojection hot path ----
        self._tt = None
        if _torch is not None:
            d = _TORCH_DEVICE
            self._tt = {
                "pos": _torch.tensor(self.sample_pos, device=d, dtype=_torch.float32),
                "vx": _torch.tensor(X, device=d, dtype=_torch.float32),
                "vy": _torch.tensor(Y, device=d, dtype=_torch.float32),
                "R": _torch.tensor(self._samp_R, device=d, dtype=_torch.float32),
                "vr": [_torch.tensor(v.real, device=d, dtype=_torch.float32) for v in self.sample_val],
                "vi": [_torch.tensor(v.imag, device=d, dtype=_torch.float32) for v in self.sample_val],
            }

    # -------------------------------------------------------------- aperture
    def _aperture(self, k: int) -> np.ndarray:
        """Fine-grid mask of scan points acquired so far (first k in raster order)."""
        k = int(np.clip(k, 0, self.scan_total))
        return (self.owner_index < k).astype(float)

    def _bp(self, holo: np.ndarray, z: float, wl: float) -> np.ndarray:
        return angular_spectrum_propagate(holo, dz=-z, wavelength=wl, dx=self.dx, dy=self.dy)

    # --------------------------------------------- backprojection reconstruction
    def _bp_intensity(self, fi: int, z: float, k: int, subtract_direct: bool = False) -> np.ndarray:
        """Backprojected intensity |I_f(r)|² at depth z from the first k walk
        samples (ALGORITHMS.md §7). I_f(r) = Σ_i val_i · exp(-i·kw·|r - pos_i|);
        subtract_direct removes the known direct emitter wave first (dark-field).
        Runs on torch/MPS when available (cos/sin, real/imag), else numpy."""
        k = int(np.clip(k, 1, self.scan_total))
        kw = 2.0 * np.pi / wavelength_of(self.freqs[fi])
        if self._tt is not None:
            return self._bp_intensity_torch(fi, float(z), k, kw, subtract_direct)
        # ----- numpy fallback -----
        P = self.sample_pos[:k]
        V = self.sample_val[fi][:k].copy()
        if subtract_direct:
            R = self._samp_R[:k]
            V = V - np.exp(1j * kw * R) / np.maximum(R, 1e-6)
        dX = self._vx[None, :, :] - P[:, 0, None, None]
        dY = self._vy[None, :, :] - P[:, 1, None, None]
        dZ = z - P[:, 2, None, None]
        dist = np.sqrt(dX * dX + dY * dY + dZ * dZ)
        img = np.tensordot(V, np.exp(-1j * kw * dist), axes=([0], [0]))
        return np.abs(img) ** 2

    def _bp_intensity_torch(self, fi, z, k, kw, subtract_direct, chunk=2000):
        """torch/MPS backprojection intensity, chunked over samples to bound memory.
        exp(-iθ) = cosθ - i·sinθ ; Re = Σ(vr·cos+vi·sin), Im = Σ(vi·cos-vr·sin)."""
        t = self._tt
        vx, vy = t["vx"], t["vy"]
        real = _torch.zeros_like(vx); imag = _torch.zeros_like(vx)
        for s in range(0, k, chunk):
            e = min(s + chunk, k)
            P = t["pos"][s:e]
            vr = t["vr"][fi][s:e].clone(); vi = t["vi"][fi][s:e].clone()
            if subtract_direct:
                R = t["R"][s:e].clamp_min(1e-6)
                vr = vr - _torch.cos(kw * R) / R
                vi = vi - _torch.sin(kw * R) / R
            dX = vx[None] - P[:, 0, None, None]
            dY = vy[None] - P[:, 1, None, None]
            dZ = z - P[:, 2, None, None]
            th = kw * _torch.sqrt(dX * dX + dY * dY + dZ * dZ)
            c = _torch.cos(th); sn = _torch.sin(th)
            real = real + (vr[:, None, None] * c + vi[:, None, None] * sn).sum(0)
            imag = imag + (vi[:, None, None] * c - vr[:, None, None] * sn).sum(0)
        return (real * real + imag * imag).cpu().numpy()

    def _stack_bp(self, mode: str, k: int) -> np.ndarray:
        nz = len(self.z_values)
        vol = np.zeros((nz, self.ny, self.nx))
        sub = (mode == "dark")
        freqs = range(len(self.freqs)) if mode == "white" else [self.carrier_idx]
        for i, z in enumerate(self.z_values):
            for fi in freqs:
                vol[i] += self._bp_intensity(fi, z, k, subtract_direct=sub)
        return vol

    # --------------------------------------------------------------- stacks
    def _stack(self, mode: str, k: int) -> np.ndarray:
        if self.backend == "backprojection":
            return self._stack_bp(mode, k)
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
        if self.backend == "backprojection":
            sub = (mode == "dark")
            freqs = range(len(self.freqs)) if mode == "white" else [self.carrier_idx]
            acc = np.zeros((self.ny, self.nx))
            for fi in freqs:
                acc += self._bp_intensity(fi, z, k, subtract_direct=sub)
            return acc
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

    def _get_vol(self, mode: str, k: int) -> np.ndarray:
        """Reconstruction volume for (mode, k). One cache BIN PER (mode, k) so
        switching between single/white/dark reuses each mode's result; cleared
        only when params change (set_params bumps _gen + resets the cache)."""
        key = (mode, int(k))
        vol = self._vol_cache.get(key)
        if vol is None:
            vol = self._stack(mode, k)
            self._vol_cache[key] = vol
            if len(self._vol_cache) > 12:           # small LRU cap
                self._vol_cache.pop(next(iter(self._vol_cache)))
        return vol

    def slice_only(self, mode: str, z_index: int, k: int, gamma: float = 0.7) -> dict:
        """Single-depth reconstruction for z-scrubbing. Reads the cached volume
        for (mode, k) if present (instant); otherwise computes just this depth."""
        z_index = int(np.clip(z_index, 0, len(self.z_values) - 1))
        z = float(self.z_values[z_index])
        vol = self._vol_cache.get((mode, int(k)))
        if vol is not None:
            img = vol[z_index]
        else:
            img = self._recon_one(mode, z, k)
        sl = (img / (img.max() or 1.0)) ** gamma
        return {"slice": _png_b64(sl, "inferno"), "z": z, "zIndex": z_index}


    # ---------------------------------------------------------------- public
    def recon(self, mode: str, z_index: int, k: int, gamma: float = 0.7) -> dict:
        vol = self._get_vol(mode, k)
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
        if self.backend == "backprojection":
            return {}                               # no uniform-plane hologram to show
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
            "backend": self.backend,
            "device": _TORCH_DEVICE or "numpy",
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

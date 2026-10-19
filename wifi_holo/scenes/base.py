"""Shared scene types and the parameter-schema helper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Built:
    """Everything the engine needs from a scene to run reconstruction."""
    # recording / reconstruction grid (metres)
    Lx: float
    Ly: float
    dx: float
    dy: float
    nx: int
    ny: int
    # depth axis (metres from the recording plane toward the scene)
    z_values: np.ndarray
    z_emitter: float                     # emitter plane (dark-field illumination)
    emitter_xy: tuple[float, float]      # lateral emitter position (emitter_pass window)
    reference_xy: tuple[float, float]    # homodyne reference antenna
    # spectrum
    freqs: np.ndarray
    carrier_idx: int
    carrier_wavelength: float
    # reconstruction backend: "angular_spectrum" (uniform-plane FFT) or
    # "backprojection" (coherent sum over scattered 3D sample positions)
    backend: str = "angular_spectrum"
    # --- angular-spectrum backend: complex field per frequency on the plane ---
    full_fields: list[np.ndarray] = field(default_factory=list)
    # --- backprojection backend: scattered 3D samples ---
    sample_pos: np.ndarray | None = None          # (N, 3) antenna positions
    sample_val: list[np.ndarray] | None = None    # per freq: (N,) complex clean field
    recon_x: np.ndarray | None = None             # (nx,) voxel x coords
    recon_y: np.ndarray | None = None             # (ny,) voxel y coords
    # metadata passed through to the frontend (3D geometry) and /api/state
    geometry: dict = field(default_factory=dict)
    scene_extra: dict = field(default_factory=dict)


def param(key, label, group, kind="slider", default=0.0, lo=0.0, hi=1.0,
          step=0.1, unit="", fmt=None, options=None, show_if=None):
    """One control descriptor for the dynamic parameter panel.

    kind: "slider" (numeric), "toggle" (boolean), or "select" (one of `options`).
    fmt:  "cm" | "m" | "ghz" | "mhz" | "int" | None -> how the frontend labels it.
    options: list of string choices (select kind only).
    show_if: {key: value} -> only show this control when another param equals value.
    """
    return {
        "key": key, "label": label, "group": group, "kind": kind,
        "default": default, "min": lo, "max": hi, "step": step,
        "unit": unit, "fmt": fmt, "options": options, "show_if": show_if,
    }


def defaults_from_schema(schema) -> dict:
    return {p["key"]: p["default"] for p in schema}

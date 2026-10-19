"""Shared scene types and the parameter-schema helper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Built:
    """Everything the engine needs from a scene to run reconstruction."""
    # recording grid (metres)
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
    # full-resolution complex field at the recording plane, one per frequency
    full_fields: list[np.ndarray]
    # metadata passed through to the frontend (3D geometry) and /api/state
    geometry: dict = field(default_factory=dict)
    scene_extra: dict = field(default_factory=dict)


def param(key, label, group, kind="slider", default=0.0, lo=0.0, hi=1.0,
          step=0.1, unit="", fmt=None):
    """One control descriptor for the dynamic parameter panel.

    kind: "slider" (numeric) or "toggle" (boolean).
    fmt:  "cm" | "m" | "ghz" | "mhz" | "int" | None -> how the frontend labels it.
    """
    return {
        "key": key, "label": label, "group": group, "kind": kind,
        "default": default, "min": lo, "max": hi, "step": step,
        "unit": unit, "fmt": fmt,
    }


def defaults_from_schema(schema) -> dict:
    return {p["key"]: p["default"] for p in schema}

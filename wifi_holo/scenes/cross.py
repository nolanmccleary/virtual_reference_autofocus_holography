"""Cross scene: the lab experiment (router illuminates a metal cross)."""

from __future__ import annotations

import numpy as np

from ..scene import CrossScene
from ..forward import record_hologram, _image_sources
from ..sources import wavelength_of
from .base import Built, param, defaults_from_schema

NAME = "cross"
LABEL = "Cross (lab experiment)"

PARAM_SCHEMA = [
    param("carrier_ghz", "Carrier", "Wave", default=5.0, lo=2.0, hi=6.0, step=0.1, fmt="ghz"),
    param("bandwidth_mhz", "Bandwidth", "Wave", default=70, lo=0, hi=200, step=5, fmt="mhz"),
    param("n_freq", "Freq. samples", "Wave", default=32, lo=1, hi=48, step=1, fmt="int"),
    param("nx", "Slice resolution", "Grid", default=120, lo=48, hi=200, step=4, fmt="px"),
    param("plane_x_m", "Plane X", "Grid", default=3.0, lo=1.0, hi=4.0, step=0.1, fmt="m"),
    param("plane_y_m", "Plane Y", "Grid", default=2.0, lo=1.0, hi=4.0, step=0.1, fmt="m"),
    param("point_sep_cm", "Point separation", "Grid", default=2.5, lo=1, hi=40, step=0.5, fmt="cm"),
    param("z_emitter_cm", "Router Z (dist)", "Router", default=230, lo=100, hi=350, step=5, fmt="cm"),
    param("emitter_x_cm", "Router X", "Router", default=150, lo=0, hi=300, step=5, fmt="cm"),
    param("emitter_y_cm", "Router Y", "Router", default=100, lo=0, hi=200, step=5, fmt="cm"),
    param("z_object_cm", "Cross Z (dist)", "Cross", default=140, lo=30, hi=330, step=5, fmt="cm"),
    param("cross_x_cm", "Cross X", "Cross", default=150, lo=0, hi=300, step=5, fmt="cm"),
    param("cross_y_cm", "Cross Y", "Cross", default=100, lo=0, hi=200, step=5, fmt="cm"),
    param("cross_arm_cm", "Cross arm", "Cross", default=100, lo=20, hi=180, step=5, fmt="cm"),
    param("cross_thickness_cm", "Cross thickness", "Cross", default=5, lo=0, hi=60, step=1, fmt="cm"),
    param("ref_x_cm", "Reference X", "Reference", default=45, lo=0, hi=300, step=5, fmt="cm"),
    param("ref_y_cm", "Reference Y", "Reference", default=30, lo=0, hi=200, step=5, fmt="cm"),
    param("multipath", "Multipath", "Multipath", kind="toggle", default=False),
    param("wall_distance_m", "Wall distance", "Multipath", default=2.5, lo=0.5, hi=6.0, step=0.1, fmt="m"),
    param("reflectivity", "Reflectivity", "Multipath", default=0.5, lo=0.0, hi=1.0, step=0.05, fmt="num"),
]


def defaults() -> dict:
    return defaults_from_schema(PARAM_SCHEMA)


def _scene(p: dict) -> CrossScene:
    g = lambda k: p.get(k, next(s["default"] for s in PARAM_SCHEMA if s["key"] == k))
    return CrossScene(
        Lx=float(g("plane_x_m")), Ly=float(g("plane_y_m")),
        dx=float(g("plane_x_m")) / int(g("nx")),
        z_object=float(g("z_object_cm")) / 100.0,
        z_emitter=float(g("z_emitter_cm")) / 100.0,
        carrier_hz=float(g("carrier_ghz")) * 1e9,
        bandwidth_hz=float(g("bandwidth_mhz")) * 1e6,
        n_frequencies=int(g("n_freq")),
        cross_arm_length=float(g("cross_arm_cm")) / 100.0,
        cross_thickness=float(g("cross_thickness_cm")) / 100.0,
        emitter_xy=(float(g("emitter_x_cm")) / 100.0, float(g("emitter_y_cm")) / 100.0),
        cross_center_xy=(float(g("cross_x_cm")) / 100.0, float(g("cross_y_cm")) / 100.0),
        reference_xy=(float(g("ref_x_cm")) / 100.0, float(g("ref_y_cm")) / 100.0),
        multipath=bool(g("multipath")),
        multipath_reflectivity=float(g("reflectivity")),
        room_margin_m=float(g("wall_distance_m")),
        room_back_m=float(g("wall_distance_m")),
        room_front_m=float(g("wall_distance_m")),
    )


def build(p: dict) -> Built:
    s = _scene(p)
    freqs = s.frequencies()
    carrier_idx = int(np.argmin(np.abs(freqs - s.carrier_hz)))
    z_values = np.linspace(0.4, s.z_emitter + 0.3, 41)
    full = [record_hologram(s, f) for f in freqs]

    geometry = {
        "kind": "cross",
        "Lx": s.Lx, "Ly": s.Ly,
        "zObject": s.z_object, "zEmitter": s.z_emitter,
        "emitterXY": list(s.emitter_xy), "refXY": list(s.reference_xy),
        "crossCenter": list(s.cross_center_xy),
        "armL": s.cross_arm_length / 2.0, "armW": s.cross_arm_width / 2.0,
        "crossThickness": s.cross_thickness,
        "occluders": [{"z": s.z_object, "kind": "cross",
                       "cx": s.cross_center_xy[0], "cy": s.cross_center_xy[1],
                       "L": s.cross_arm_length / 2.0, "w": s.cross_arm_width / 2.0}],
        "multipath": bool(s.multipath),
        "room": {
            "x0": -s.room_margin_m, "x1": s.Lx + s.room_margin_m,
            "y0": -s.room_margin_m, "y1": s.Ly + s.room_margin_m,
            "zFront": -s.room_front_m, "zBack": s.z_emitter + s.room_back_m,
        },
    }
    return Built(
        Lx=s.Lx, Ly=s.Ly, dx=s.dx, dy=s.dy, nx=s.nx, ny=s.ny,
        z_values=z_values, z_emitter=s.z_emitter,
        emitter_xy=s.emitter_xy, reference_xy=s.reference_xy,
        freqs=freqs, carrier_idx=carrier_idx, carrier_wavelength=s.carrier_wavelength,
        full_fields=full, geometry=geometry,
        scene_extra={"iObject": int(np.argmin(np.abs(z_values - s.z_object)))},
    )

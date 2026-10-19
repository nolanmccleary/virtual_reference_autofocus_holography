"""Room scene: the simulated storage building of PRL 118, 183901 Fig 4(e).

Internal convention matches the cross scene: the recording plane (the ceiling,
a 2D antenna array) is at z = 0; depth z increases downward into the building
toward the dipole emitter on the floor at z = height. Reflective shelves are
absorbing slabs at intermediate heights; the building walls generate multipath
via first-order image sources.
"""

from __future__ import annotations

import numpy as np

from ..propagation import angular_spectrum_propagate as _asp
from ..sources import point_source_field, wavelength_of
from .base import Built, param, defaults_from_schema

NAME = "room"
LABEL = "Room (storage building, Fig 4e)"

PARAM_SCHEMA = [
    param("carrier_ghz", "Carrier", "Wave", default=2.4, lo=2.0, hi=6.0, step=0.1, fmt="ghz"),
    param("bandwidth_mhz", "Bandwidth", "Wave", default=20, lo=0, hi=200, step=5, fmt="mhz"),
    param("n_freq", "Freq. samples", "Wave", default=16, lo=1, hi=32, step=1, fmt="int"),
    param("nx", "Slice resolution", "Grid", default=200, lo=96, hi=280, step=8, fmt="px"),
    param("building_x_m", "Building X", "Building", default=20.0, lo=8.0, hi=28.0, step=0.5, fmt="m"),
    param("building_y_m", "Building Y", "Building", default=17.0, lo=8.0, hi=24.0, step=0.5, fmt="m"),
    param("height_m", "Ceiling height", "Building", default=12.0, lo=5.0, hi=16.0, step=0.5, fmt="m"),
    param("point_sep_cm", "Point separation", "Grid", default=15, lo=5, hi=60, step=1, fmt="cm"),
    param("emitter_x_m", "Emitter X", "Emitter", default=14.0, lo=0.0, hi=28.0, step=0.5, fmt="m"),
    param("emitter_y_m", "Emitter Y", "Emitter", default=8.5, lo=0.0, hi=24.0, step=0.5, fmt="m"),
    param("shelf_height_m", "Shelf height", "Shelves", default=6.0, lo=1.0, hi=11.0, step=0.5, fmt="m"),
    param("multipath", "Multipath (walls)", "Multipath", kind="toggle", default=True),
    param("reflectivity", "Reflectivity", "Multipath", default=0.4, lo=0.0, hi=1.0, step=0.05, fmt="num"),
]


def defaults() -> dict:
    return defaults_from_schema(PARAM_SCHEMA)


def _g(p, k):
    return p.get(k, next(s["default"] for s in PARAM_SCHEMA if s["key"] == k))


def _shelves(Lx, Ly, h):
    """Rectangular absorbing shelves (x0,y0,x1,y1) at two heights, as fractions
    of the footprint. The emitter sits in the open area not under a shelf."""
    hi = h                       # upper mezzanine / shelf plane
    lo = h * 0.5                 # lower shelf plane
    return [
        (0.10 * Lx, 0.15 * Ly, 0.48 * Lx, 0.85 * Ly, hi),   # big mezzanine block
        (0.55 * Lx, 0.55 * Ly, 0.90 * Lx, 0.90 * Ly, lo),   # lower shelf A
        (0.55 * Lx, 0.10 * Ly, 0.80 * Lx, 0.40 * Ly, lo),   # lower shelf B
    ]


def _mask_for_plane(X, Y, rects):
    t = np.ones(X.shape)
    for (x0, y0, x1, y1) in rects:
        t[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = 0.0
    return t


def _record(X, Y, dx, dy, src_xyz, height, shelves, wl):
    """Propagate one source from the floor up through the shelves to the ceiling."""
    # group shelves by z-plane, process from farthest (near floor) to ceiling
    planes = sorted({s[4] for s in shelves}, reverse=True)
    u = point_source_field(X, Y, planes[0] if planes else 0.0, src_xyz, wl)
    for i, zp in enumerate(planes):
        rects = [(s[0], s[1], s[2], s[3]) for s in shelves if s[4] == zp]
        u = u * _mask_for_plane(X, Y, rects)
        next_z = planes[i + 1] if i + 1 < len(planes) else 0.0
        u = _asp(u, dz=(zp - next_z), wavelength=wl, dx=dx, dy=dy)
    if not planes:
        u = _asp(point_source_field(X, Y, height, src_xyz, wl), dz=height,
                 wavelength=wl, dx=dx, dy=dy)
    return u


def build(p: dict) -> Built:
    Lx = float(_g(p, "building_x_m"))
    Ly = float(_g(p, "building_y_m"))
    h = float(_g(p, "height_m"))
    nx = int(_g(p, "nx"))
    dx = Lx / nx
    ny = max(1, int(round(Ly / dx)))
    dy = Ly / ny

    carrier = float(_g(p, "carrier_ghz")) * 1e9
    bw = float(_g(p, "bandwidth_mhz")) * 1e6
    nf = int(_g(p, "n_freq"))
    freqs = (np.array([carrier]) if nf <= 1
             else np.linspace(carrier - bw / 2, carrier + bw / 2, nf))
    carrier_idx = int(np.argmin(np.abs(freqs - carrier)))

    ex, ey = float(_g(p, "emitter_x_m")), float(_g(p, "emitter_y_m"))
    emitter_xyz = (ex, ey, h)
    shelves = _shelves(Lx, Ly, float(_g(p, "shelf_height_m")))
    refl = float(_g(p, "reflectivity"))
    multipath = bool(_g(p, "multipath"))

    # image sources: mirror the emitter across the 4 building walls + ceiling
    images = []
    if multipath:
        images = [
            ((-ex, ey, h), refl), ((2 * Lx - ex, ey, h), refl),
            ((ex, -ey, h), refl), ((ex, 2 * Ly - ey, h), refl),
            ((ex, ey, -h), refl),              # reflection across the ceiling plane
        ]

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    full = []
    for f in freqs:
        wl = wavelength_of(f)
        u = _record(X, Y, dx, dy, emitter_xyz, h, shelves, wl)
        for pos, amp in images:                # reflected paths skip the shelves
            u = u + amp * point_source_field(X, Y, 0.0, pos, wl)
        full.append(u)

    # reference antenna: a ceiling corner
    ref_xy = (0.08 * Lx, 0.08 * Ly)
    # depth axis: finer than cross (deep building) for emitter localization
    z_values = np.linspace(0.3, h + 0.6, 61)

    geometry = {
        "kind": "room",
        "Lx": Lx, "Ly": Ly, "height": h,
        "zEmitter": h, "emitterXYZ": [ex, ey, h],
        "refXY": list(ref_xy),
        "shelves": [list(s) for s in shelves],
        "multipath": multipath,
    }
    return Built(
        Lx=Lx, Ly=Ly, dx=dx, dy=dy, nx=nx, ny=ny,
        z_values=z_values, z_emitter=h,
        emitter_xy=(ex, ey), reference_xy=ref_xy,
        freqs=freqs, carrier_idx=carrier_idx, carrier_wavelength=wavelength_of(carrier),
        full_fields=full, geometry=geometry,
        scene_extra={"iEmitter": int(np.argmin(np.abs(z_values - h)))},
    )

"""Forward model: synthesise the complex hologram recorded on the scan plane.

Pipeline for a single frequency:
  1. Launch the emitter's diverging spherical wave analytically onto the object
     plane (exact free-space Green's function).
  2. Multiply by the object's amplitude transmission (the cross).
  3. Angular-spectrum propagate from the object plane to the recording plane.

The direct (unobstructed) illumination is included automatically: the
transmission is 1 everywhere outside the opaque cross, so the propagated field
is direct-light + shadow.
"""

from __future__ import annotations

import numpy as np

from .propagation import angular_spectrum_propagate
from .scene import CrossScene
from .sources import point_source_field, wavelength_of, box_image_sources


def _direct_field(scene: CrossScene, source_xyz, wavelength: float) -> np.ndarray:
    """Field at the recorder from a source, through the (finite-thickness) object.

    A finite-thickness cross is modelled as a short stack of thin absorbing masks
    spanning [z_object - T/2, z_object + T/2]; thickness 0 reduces to one mask.
    """
    X, Y = scene.grids()
    mask = scene.cross_mask()
    T = scene.cross_thickness
    if T <= 0:
        planes = [scene.z_object]
    else:
        n = int(np.clip(round(T / 0.05) + 1, 2, 6))
        planes = list(np.linspace(scene.z_object + T / 2.0, scene.z_object - T / 2.0, n))

    u = point_source_field(X, Y, planes[0], source_xyz, wavelength)
    for i, zp in enumerate(planes):
        u = u * mask
        next_z = planes[i + 1] if i + 1 < len(planes) else 0.0
        u = angular_spectrum_propagate(
            u, dz=(zp - next_z), wavelength=wavelength, dx=scene.dx, dy=scene.dy)
    return u


def _image_sources(scene: CrossScene):
    """Cross-scene wall image sources via the SHARED box-room helper. Walls sit
    `room_margin_m` beyond the scan plane (and back/front beyond emitter/recorder)."""
    m = scene.room_margin_m
    box = (-m, scene.Lx + m, -m, scene.Ly + m,
           -scene.room_front_m, scene.z_emitter + scene.room_back_m)
    return box_image_sources(scene.emitter_xyz, box, scene.multipath_reflectivity)


def record_hologram(scene: CrossScene, frequency_hz: float) -> np.ndarray:
    """Complex field recorded on the scan plane (z=0) for one frequency.

    Direct wave (through the object) plus, when multipath is enabled, unstructured
    reflected waves from wall image sources -> standing waves / speckle.
    """
    wavelength = wavelength_of(frequency_hz)
    u = _direct_field(scene, scene.emitter_xyz, wavelength)
    if scene.multipath:
        X, Y = scene.grids()
        for pos, amp in _image_sources(scene):
            u = u + amp * point_source_field(X, Y, 0.0, pos, wavelength)
    return u


def reference_value(scene: CrossScene, frequency_hz: float) -> complex:
    """Complex field the stationary reference antenna sees, for homodyne norm.

    Sampled from the recorded field at the reference antenna location.
    """
    u_rec = record_hologram(scene, frequency_hz)
    rx, ry = scene.reference_xy
    ix = int(round(rx / scene.dx - 0.5))
    iy = int(round(ry / scene.dy - 0.5))
    ix = int(np.clip(ix, 0, scene.nx - 1))
    iy = int(np.clip(iy, 0, scene.ny - 1))
    return complex(u_rec[iy, ix])


def direct_field_at_recorder(scene: CrossScene, frequency_hz: float) -> np.ndarray:
    """Analytic spherical wave at the recorder with NO object (validation ref)."""
    wavelength = wavelength_of(frequency_hz)
    X, Y = scene.grids()
    return point_source_field(X, Y, 0.0, scene.emitter_xyz, wavelength)

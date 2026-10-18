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
from .sources import point_source_field, wavelength_of


def record_hologram(scene: CrossScene, frequency_hz: float) -> np.ndarray:
    """Complex field recorded on the scan plane (z=0) for one frequency."""
    wavelength = wavelength_of(frequency_hz)
    X, Y = scene.grids()

    # 1. emitter -> object plane (analytic spherical wave)
    u_obj = point_source_field(X, Y, scene.z_object, scene.emitter_xyz, wavelength)

    # 2. apply object transmission
    u_obj = u_obj * scene.cross_mask()

    # 3. object plane -> recording plane (toward the recorder, distance z_object)
    #    The wave travels away from the source, so dz = +z_object.
    u_rec = angular_spectrum_propagate(
        u_obj, dz=scene.z_object, wavelength=wavelength, dx=scene.dx, dy=scene.dy
    )
    return u_rec


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

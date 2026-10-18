"""wifi_holo: reproduction of Holl & Reinhard, 'Holography of Wi-fi Radiation',
PRL 118, 183901 (2017).

Scalar angular-spectrum wave engine for forward hologram synthesis and
digital reconstruction (backpropagation, white-light, dark-field).
"""

from .propagation import angular_spectrum_propagate
from .sources import point_source_field, speed_of_light, wavelength_of
from .scene import CrossScene

__all__ = [
    "angular_spectrum_propagate",
    "point_source_field",
    "speed_of_light",
    "wavelength_of",
    "CrossScene",
]

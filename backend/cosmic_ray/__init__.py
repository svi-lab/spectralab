"""Cosmic-ray removal: :class:`CosmicRayRemover` and helpers.

File layout
-----------
Public API (re-exported here):
  _remover.py  — CosmicRayRemover class (routing + high-level interface)
  mask_1d.py   — 1D spike detection and repair
  mask_map.py  — spatial (2D/3D) detection and repair
  harmonic.py  — Nd:YAG laser-harmonic and grating 2nd-order artifact notch removal
  _units.py    — nm/cm^-1 <-> channel width conversion (median_channel_width,
                 spectral_to_channels are public despite the leading underscore
                 on the module; the frontend uses them to show physical CR
                 widths without duplicating the conversion)

Private implementation details (not imported outside this package):
  _mad.py  — scaled-MAD noise estimator used by mask_1d and mask_map
"""

from ._remover import CosmicRayRemover
from ._units import median_channel_width, spectral_to_channels
from .mask_1d import remove_cosmic_rays_1d

__all__ = [
    "CosmicRayRemover",
    "median_channel_width",
    "remove_cosmic_rays_1d",
    "spectral_to_channels",
]

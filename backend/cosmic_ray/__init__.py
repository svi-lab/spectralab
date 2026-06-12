# -*- coding: utf-8 -*-
"""Cosmic-ray removal: :class:`CosmicRayRemover` and helpers.

File layout
-----------
Public API (re-exported here):
  _remover.py  — CosmicRayRemover class (routing + high-level interface)
  mask_1d.py   — 1D spike detection and repair
  mask_map.py  — spatial (2D/3D) detection and repair
  harmonic.py  — Nd:YAG laser-harmonic notch removal

Private implementation details (not imported outside this package):
  _1d.py   — low-level 1D medfilt/MAD routines used by mask_1d
  _mad.py  — scaled-MAD noise estimator used by mask_1d and mask_map
"""

from .mask_1d import remove_cosmic_rays_1d
from ._remover import CosmicRayRemover

__all__ = ["CosmicRayRemover", "remove_cosmic_rays_1d"]

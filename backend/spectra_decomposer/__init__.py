# -*- coding: utf-8 -*-
"""NMF-based spectral pattern decomposition: :class:`Decomposer`."""

from ._decomposer import Decomposer
from ._nmf import compute_nmf_diagnostic_curve

__all__ = ["Decomposer", "compute_nmf_diagnostic_curve"]

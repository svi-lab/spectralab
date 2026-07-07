# -*- coding: utf-8 -*-
"""MCR-ALS curve resolution: pure-component spectra + concentration maps."""

from ._mcr import compute_mcr_ambiguity, compute_mcr_rank_svd, mcr_als
from ._mcr_decomposer import MCRDecomposer

__all__ = [
    "MCRDecomposer",
    "compute_mcr_rank_svd",
    "compute_mcr_ambiguity",
    "mcr_als",
]

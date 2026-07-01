# -*- coding: utf-8 -*-
"""Gaussian peak deconvolution: :class:`PeakFitter` and :func:`fit_map_gaussian`."""

from ._batch import BatchFitResult, fit_map_gaussian
from ._fitter import BandResult, BandSpec, FitResult, PeakFitter

__all__ = [
    "PeakFitter",
    "BandSpec",
    "BandResult",
    "FitResult",
    "fit_map_gaussian",
    "BatchFitResult",
]

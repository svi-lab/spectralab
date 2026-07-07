# -*- coding: utf-8 -*-
"""Gaussian peak deconvolution: :class:`PeakFitter` and :func:`fit_map_gaussian`."""

from ._batch import BatchFitResult, fit_map_gaussian
from ._fitter import BandResult, BandSpec, FitResult, PeakFitter
from ._presets import BandPreset, get_preset_bands, list_preset_materials

__all__ = [
    "PeakFitter",
    "BandSpec",
    "BandResult",
    "FitResult",
    "fit_map_gaussian",
    "BatchFitResult",
    "BandPreset",
    "get_preset_bands",
    "list_preset_materials",
]

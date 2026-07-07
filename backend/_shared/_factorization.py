# -*- coding: utf-8 -*-
"""Shared plumbing for population factorization methods (NMF, MCR-ALS).

Both factorization backends reshape a ``(..., n_spectral)`` cube into a
``(n_spectra, n_spectral)`` row-stack, handle CleanData's all-NaN rows, and
prepare a strictly non-negative, min-subtracted matrix to fit on. That
preparation is identical for NMF (sklearn rejects any negative entry) and
MCR-ALS (NNLS requires non-negative input / initial estimates), so it lives
here as one source of truth rather than being duplicated per package.
"""

from __future__ import annotations

import numpy as np


def _flatten_to_row_stack(
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape ``(..., n_spectral)`` to ``(n_spectra, n_spectral)``.

    Returns the row-stack and the original spatial shape (everything before
    the spectral axis), so output can be reshaped back.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2:
        raise ValueError(
            "Factorization needs >= 2D input (last axis = spectral); "
            f"got ndim={arr.ndim}"
        )
    spatial_shape = arr.shape[:-1]
    n_spectral = arr.shape[-1]
    return arr.reshape(-1, n_spectral), spatial_shape


def _nonnegative_fit_matrix(
    row_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shift each valid row to a minimum of exactly 0, fill all-NaN rows with
    the column-median of valid rows, and clip floating-point round-off.

    Non-negative factorization rejects negative input outright (NMF raises,
    NNLS is only defined for non-negative estimates) — the per-spectrum
    min-subtraction below is therefore not optional the way PCA's
    ``subtract_min`` is.

    Returns
    -------
    spectra_for_fit, per_spec_min, nan_row_mask, valid_idx
    """
    n_spectra, n_spectral = row_stack.shape
    nan_row_mask = np.all(np.isnan(row_stack), axis=1)
    valid_idx = ~nan_row_mask

    per_spec_min = np.zeros((n_spectra, 1), dtype=float)
    if valid_idx.any():
        per_spec_min[valid_idx] = row_stack[valid_idx].min(axis=-1, keepdims=True)
    spectra_for_fit = row_stack - per_spec_min

    if nan_row_mask.any():
        valid_fit = spectra_for_fit[valid_idx]
        fill = (
            np.nanmedian(valid_fit, axis=0)
            if valid_fit.shape[0] > 0
            else np.zeros(n_spectral)
        )
        spectra_for_fit = spectra_for_fit.copy()
        spectra_for_fit[nan_row_mask] = fill

    # Min-subtraction makes every valid row's true minimum exactly 0, but
    # floating-point round-off can still leave e.g. -1e-16 — non-negative
    # factorizers reject any negative entry, so clip it away.
    spectra_for_fit = np.clip(spectra_for_fit, 0, None)
    return spectra_for_fit, per_spec_min, nan_row_mask, valid_idx


__all__ = ["_flatten_to_row_stack", "_nonnegative_fit_matrix"]

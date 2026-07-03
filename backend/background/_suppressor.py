# -*- coding: utf-8 -*-
"""BackgroundSuppressor: subtract a scaled substrate reference from each spectrum.

Teaching note — what this class does and why:

    Measured PL spectrum ≈ film emission + c · substrate emission
    => corrected = measured - c · reference

    `c` is a fixed physics-predicted scale (c_physics × power/exposure ratio),
    computed on the Preprocessing page from the Sample Structure summary.

NaN-row handling mirrors backend/spectra_cleaner/_cleaner.py (the house pattern):
    detect NaN rows upfront → skip them → restore NaN after subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from _shared._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from _shared.utils import ensure_in_memory

_TREATMENT_KEY = "background_suppression"


@dataclass
class BackgroundSuppressor:
    """Subtract a scaled substrate reference spectrum from PL data.

    Parameters
    ----------
    reference : ndarray, shape (n_spectral,)
        Reference spectrum on the *same spectral axis* as the data.
        If axes differ, interpolate with _scale.interp_reference before
        constructing this object.
    spectral_dim : str or None
        Override spectral dimension name (auto-detected if None).
    fixed_scale : float
        Physics-predicted c for every spectrum (c_physics × power/exposure ratio).
        Only valid in raw / power-corrected intensity space — the caller must
        ensure no data-dependent normalization (min-max/area) was applied to
        either the data or the reference before this point.
    """

    reference: np.ndarray
    fixed_scale: float
    spectral_dim: str | None = None

    def suppress(self, spectra: xr.DataArray) -> tuple[xr.DataArray, dict[str, Any]]:
        """Return (corrected DataArray, meta dict).

        meta keys:
          c_median    — median fitted c across valid spectra (scalar, for QA)
          c_values    — per-spectrum c array in original spatial layout (or scalar)
          n_nan_rows  — number of all-NaN rows skipped
          scale_mode  — always "fixed"
        """
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(f"BackgroundSuppressor expects xr.DataArray; got {type(spectra).__name__}")

        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)
        da_w = ensure_in_memory(da_w, caller="BackgroundSuppressor",
                                reason="background subtraction needs the full array in memory.")

        # ── Flatten to row stack ──────────────────────────────────────────
        spatial_shape = da_w.shape[:-1]
        n_spectral = da_w.shape[-1]
        row_stack = da_w.values.reshape(-1, n_spectral)

        # ── Detect NaN rows (CleanData dead pixels) ───────────────────────
        nan_row_mask = np.all(np.isnan(row_stack), axis=1)
        valid_idx = ~nan_row_mask
        valid_rows = row_stack[valid_idx]

        ref = np.asarray(self.reference, dtype=float)
        if len(ref) != n_spectral:
            raise ValueError(
                f"Reference length ({len(ref)}) != spectral channels ({n_spectral}). "
                "Interpolate the reference onto the data axis before suppressing."
            )

        # ── Fixed physics scale ───────────────────────────────────────────
        c_before = np.full(valid_rows.shape[0], float(self.fixed_scale))

        # ── Subtract and clip ─────────────────────────────────────────────
        corrected_valid = valid_rows - c_before[:, np.newaxis] * ref[np.newaxis, :]
        # PL intensity is non-negative; clip to avoid numerical artefacts from imperfect c.
        corrected_valid = np.clip(corrected_valid, 0.0, None)
        corrected_rows = np.full_like(row_stack, np.nan)
        corrected_rows[valid_idx] = corrected_valid

        # ── Reshape back ──────────────────────────────────────────────────
        corrected_w = reshape_row_stack_to(corrected_rows, da_w.shape)
        if tuple(da_w.dims) != orig_order:
            corrected_da = da_w.copy(data=corrected_w).transpose(*orig_order)
            corrected = corrected_da.values
        else:
            corrected = corrected_w

        # Rebuild c in spatial layout (NaN for dead rows)
        c_spatial = np.full(row_stack.shape[0], np.nan)
        c_spatial[valid_idx] = c_before
        c_spatial_reshaped = c_spatial.reshape(spatial_shape) if len(spatial_shape) > 1 else c_spatial

        meta: dict[str, Any] = {
            "c_median":    float(np.nanmedian(c_before)) if len(c_before) > 0 else float("nan"),
            "c_values":    c_spatial_reshaped,
            "n_nan_rows":  int(nan_row_mask.sum()),
            "scale_mode":  "fixed",
        }

        # Only put serialisable scalars in DataArray attrs
        attrs_meta = {k: v for k, v in meta.items()
                      if not isinstance(v, np.ndarray)}

        out_da = with_new_values(spectra, corrected, _TREATMENT_KEY, attrs_meta)
        return out_da, meta


__all__ = ["BackgroundSuppressor"]

# -*- coding: utf-8 -*-
"""Pure-numpy helpers for background suppression."""

from __future__ import annotations

import numpy as np


def interp_reference(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    target_x: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Interpolate reference onto target spectral axis.

    Out-of-range channels are set to 0 (no extrapolation).
    Returns (interpolated_ref, meta) where meta reports overlap diagnostics.
    """
    ref_y_interp = np.interp(target_x, ref_x, ref_y, left=0.0, right=0.0)

    overlap_lo = max(ref_x.min(), target_x.min())
    overlap_hi = min(ref_x.max(), target_x.max())
    if overlap_hi > overlap_lo:
        overlap_mask = (target_x >= overlap_lo) & (target_x <= overlap_hi)
        overlap_fraction = float(overlap_mask.sum()) / len(target_x)
    else:
        overlap_fraction = 0.0

    n_outside = int((target_x < ref_x.min()).sum() + (target_x > ref_x.max()).sum())

    return ref_y_interp, {"overlap_fraction": overlap_fraction, "n_outside": n_outside}


__all__ = [
    "interp_reference",
]

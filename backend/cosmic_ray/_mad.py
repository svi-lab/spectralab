"""Robust noise scale (MAD) helpers for cosmic-ray detection."""

from __future__ import annotations

import warnings

import numpy as np

# Scales MAD to a nominal Gaussian standard deviation for thresholding.
_SCALED_MAD_TO_GAUSSIAN_SIGMA = 1.4826


def scaled_median_absolute_deviation_noise(x: np.ndarray) -> float:
    """``1.4826 * median(|x - median(x)|)`` — robust spread of ``x``."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return _SCALED_MAD_TO_GAUSSIAN_SIGMA * mad


def noise_estimate_too_small(
    noise: float,
    reference_scale: float,
) -> bool:
    """True if ``noise`` is too small or non-finite for stable
    thresholding."""
    if not np.isfinite(noise):
        return True
    floor = 1e-15 * (reference_scale + np.finfo(float).tiny)
    return noise <= 0.0 or noise < floor


def robust_mad_noise_with_floor(
    deviations: np.ndarray,
    amplitude_reference: float,
) -> float:
    """Scaled MAD of ``deviations``; if tiny, bump using
    ``amplitude_reference``."""
    noise = scaled_median_absolute_deviation_noise(deviations)
    ref = float(amplitude_reference) + np.finfo(float).tiny
    if noise < 1e-15 * ref:
        noise = max(noise, 1e-12 * ref)
    return noise


# A local block whose own MAD collapses (a perfectly flat synthetic stretch)
# still gets this fraction of the whole-spectrum noise as a floor, so the
# detector cannot become infinitely sensitive there.
_LOCAL_NOISE_GLOBAL_FLOOR_FRACTION: float = 0.1


def local_mad_noise(
    deviations: np.ndarray,
    window: int,
    *,
    amplitude_reference: float,
) -> np.ndarray:
    """Per-channel noise scale: scaled MAD measured in blocks of ``window``.

    Shot-noise-limited detectors have ``sigma ∝ sqrt(intensity)``, so a single
    MAD over a spectrum spanning two decades of counts is simultaneously too
    strict in the dim regions and too loose in the bright ones.  This measures
    the spread locally instead.

    Blocks are non-overlapping (a rolling median over ~10k channels × ~800
    spectra is far too slow); the per-block values are then linearly
    interpolated from the block centres onto every channel, which is smooth
    enough for thresholding.  NaN-safe: all-NaN blocks fall back to the
    whole-spectrum value.

    Returns an array shaped like ``deviations``.
    """
    dev = np.asarray(deviations, dtype=float)
    n = dev.size
    global_noise = robust_mad_noise_with_floor(
        dev[np.isfinite(dev)] if np.any(np.isfinite(dev)) else dev,
        amplitude_reference,
    )
    window = int(max(window, 3))
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= window:
        return np.full(n, global_noise, dtype=float)

    n_blocks = max(int(np.ceil(n / window)), 2)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    centres = np.empty(n_blocks, dtype=float)
    values = np.empty(n_blocks, dtype=float)
    with warnings.catch_warnings():
        # An all-NaN block is expected (CleanData / exclusion gaps).
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(n_blocks):
            lo, hi = edges[b], edges[b + 1]
            centres[b] = 0.5 * (lo + hi - 1)
            block = dev[lo:hi]
            med = np.nanmedian(block)
            mad = np.nanmedian(np.abs(block - med))
            values[b] = _SCALED_MAD_TO_GAUSSIAN_SIGMA * mad

    bad = ~np.isfinite(values)
    if np.all(bad):
        return np.full(n, global_noise, dtype=float)
    if np.any(bad):
        values[bad] = np.interp(centres[bad], centres[~bad], values[~bad])

    noise = np.interp(np.arange(n, dtype=float), centres, values)
    return np.maximum(noise, _LOCAL_NOISE_GLOBAL_FLOOR_FRACTION * global_noise)

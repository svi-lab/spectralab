"""1D spectrum cosmic-ray (positive spike) detection and repair."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import medfilt

from ._mad import noise_estimate_too_small, robust_mad_noise_with_floor

# SNIP iterations used for repair when broad detection is disabled.
_SNIP_REPAIR_ITERATIONS_DEFAULT: int = 15

# Residual vs the median reference must also exceed this fraction of the
# local continuum.  On a smooth PL band the MAD of (y − medfilt) is ~0, so
# a threshold × MAD gate alone flags the convex peak cap as a cosmic ray.
# A 15% floor leaves typical PL bands (FWHM ≳ 80 channels) as spectral
# shape; a bright CR on that band is still several times the continuum.
_MIN_RELATIVE_RESIDUAL: float = 0.15


# ---------------------------------------------------------------------------
# SNIP background estimator
# ---------------------------------------------------------------------------


def snip_background_1d(y: np.ndarray, n_iterations: int) -> np.ndarray:
    """Background via SNIP (Sensitive Nonlinear Iterative Peak-clipping).

    Morháč et al., Nucl. Instr. Methods A 401 (1997) 113–132.

    The result is always ≤ y (for non-negative input) and does not contain
    any positive spike with half-width ≤ ``n_iterations`` channels.  Setting
    ``n_iterations`` to ~half the broadest expected CR width is sufficient.
    """
    v = np.log(np.log(np.sqrt(np.maximum(y, 0.0) + 1.0) + 1.0) + 1.0)
    for m in range(1, n_iterations + 1):
        v_new = v.copy()
        v_new[m:-m] = np.minimum(v[m:-m], (v[: -2 * m] + v[2 * m :]) / 2)
        v = v_new
    bg = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0
    return np.maximum(bg, 0.0)


# ---------------------------------------------------------------------------
# Non-destructive repair
# ---------------------------------------------------------------------------


def repair_masked_channels_1d(
    y: np.ndarray,
    mask: np.ndarray,
    bg_snip: np.ndarray,
) -> np.ndarray:
    """Fill masked channels using the better of linear interpolation and
    the SNIP background.

    For narrow gaps on a flat baseline, linear interpolation from the
    nearest good channels is most accurate.  For broad gaps that span a
    spectral feature (e.g. a PL peak), the SNIP background tracks the
    underlying feature and prevents the repair from dipping below it.

    The result is clamped to ≥ 0 (PL / Raman intensities are non-negative).
    """
    if not np.any(mask):
        return y.copy()
    good = ~mask
    if not np.any(good):
        return y.copy()

    bad_idx = np.flatnonzero(mask)
    good_idx = np.flatnonzero(good)
    x = np.arange(len(y), dtype=float)

    if good_idx.size == 1:
        linear = y.copy()
        linear[bad_idx] = y[good_idx[0]]
    else:
        linear = y.copy()
        linear[bad_idx] = np.interp(x[bad_idx], x[good_idx], y[good_idx])

    # max(linear, snip_bg): linear wins for narrow gaps on a flat baseline;
    # snip_bg wins when linear would dip below the true underlying feature.
    repaired = np.where(mask, np.maximum(linear, bg_snip), y)
    return np.maximum(repaired, 0.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_float_1d_spectrum(y: np.ndarray, kernel_size: int) -> np.ndarray:
    """Cast y to float 1D; validate kernel_size is odd and ≥ 3."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {arr.shape}")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"spike_width must be odd and >= 3, got {kernel_size}")
    return arr


def _dilate_mask_1d(mask: np.ndarray) -> np.ndarray:
    """Expand a boolean mask by 1 channel on each side."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    return out


def _zero_saturation_mask(y: np.ndarray) -> np.ndarray:
    """Flag near-zero channels surrounded by positive neighbours.

    Detects ADC-saturation artefacts where the detector clips to 0.
    Channel i is flagged when y[i] < 1e-4 × median(positive) AND at least
    2 of its 4 nearest neighbours exceed 10% of that median.
    """
    pos = y[y > 0]
    if pos.size < 3:
        return np.zeros(y.size, dtype=bool)
    pos_median = float(np.median(pos))
    floor = 1e-4 * pos_median
    nbr_thr = 0.1 * pos_median
    near_zero = y <= floor
    is_pos = (y > nbr_thr).astype(np.int8)
    padded = np.pad(is_pos, 2, mode="edge")
    nbr_sum = padded[:-4] + padded[1:-3] + padded[3:-1] + padded[4:]
    return near_zero & (nbr_sum >= 2)


# ---------------------------------------------------------------------------
# Detection primitives (also used by other modules)
# ---------------------------------------------------------------------------


def positive_spike_mask_vs_median_smooth(
    y: np.ndarray,
    median_smoothed_y: np.ndarray,
    threshold_multiplier: float,
) -> tuple[np.ndarray, float]:
    """Mask where positive residual exceeds the MAD gate *and* a relative floor.

    Residual is y − median_smoothed_y; noise is scaled MAD of residual.
    A cosmic ray must also exceed ``_MIN_RELATIVE_RESIDUAL`` × the local
    median-filtered intensity, otherwise a smooth PL peak cap is flagged
    whenever MAD collapses toward zero.
    """
    residual = y - median_smoothed_y
    if not np.any(residual):
        return np.zeros(y.shape, dtype=bool), 0.0
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(median_smoothed_y))),
    )
    noise = robust_mad_noise_with_floor(residual, amplitude_reference)
    rel_floor = _MIN_RELATIVE_RESIDUAL * np.maximum(median_smoothed_y, 0.0)
    mask = residual > np.maximum(threshold_multiplier * noise, rel_floor)
    return mask.astype(bool), noise


def positive_spike_mask_from_derivative_peaks(
    y: np.ndarray,
    threshold_multiplier: float,
) -> np.ndarray:
    """Interior i where y[i] is above both neighbours by
    threshold_multiplier × noise.

    noise is scaled MAD of diff(y).
    """
    dy = np.diff(y)
    n = y.size
    mask = np.zeros(n, dtype=bool)
    if dy.size == 0:
        return mask
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(dy))),
    )
    noise = robust_mad_noise_with_floor(dy, amplitude_reference)
    max_abs_dy = float(np.nanmax(np.abs(dy))) + np.finfo(float).tiny
    if noise_estimate_too_small(noise, max_abs_dy):
        return mask
    threshold = threshold_multiplier * noise
    for i in range(1, n - 1):
        if (y[i] - y[i - 1] > threshold) and (y[i] - y[i + 1] > threshold):
            mask[i] = True
    return mask


def linear_interpolate_masked_channels_1d(
    y: np.ndarray,
    bad_channel_mask: np.ndarray,
) -> np.ndarray:
    """Fill masked channels by linear interpolation from good ones.

    Kept as a utility for the harmonic notch and map engines.
    For the main 1D CR repair pipeline, use repair_masked_channels_1d.
    """
    if not np.any(bad_channel_mask):
        return y.copy()
    good = ~bad_channel_mask
    if not np.any(good):
        return y.copy()
    n = y.size
    x = np.arange(n, dtype=float)
    out = y.copy()
    bad_idx = np.flatnonzero(bad_channel_mask)
    good_idx = np.flatnonzero(good)
    if good_idx.size == 1:
        out[bad_idx] = out[good_idx[0]]
        return out
    out[bad_idx] = np.interp(x[bad_idx], x[good_idx], y[good_idx])
    return out


# ---------------------------------------------------------------------------
# Main 1D removal function
# ---------------------------------------------------------------------------


def remove_cosmic_rays_1d(
    y: np.ndarray,
    *,
    kernel_size: int = 5,
    threshold: float = 5.0,
    max_passes: int = 3,
    broad_spike_width: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove sharp and broad positive spikes from a single 1D spectrum.

    **Narrow pass** (up to ``max_passes`` iterations): medfilt reference with
    ``kernel_size``.  Catches spikes narrower than ``kernel_size // 2``
    channels.

    **Broad pass** (one pass, if ``broad_spike_width > 0``): medfilt reference
    with ``kernel_size = 2 × broad_spike_width + 1``.  Catches spikes narrower
    than ``broad_spike_width`` channels.  Run on the already narrow-corrected
    signal so narrow spikes do not contaminate the broad reference.

    **Repair**: uses the SNIP background as a floor — prevents the corrected
    signal from dipping below the true spectral baseline (negative-peak
    artefact common with pure linear interpolation across broad masked regions
    on sloped features).

    Parameters
    ----------
    y
        One spectral trace (any numeric dtype; cast to float).
    kernel_size
        Odd ≥ 3; medfilt window for the narrow pass.  Increase for slightly
        wider narrow spikes (e.g. 9–13 for 4–6 channel CRs).
    threshold
        MAD multiplier.  Lower → more detections.  Default 5.0 (use 3.0–4.0
        for clean, low-noise data).
    max_passes
        Maximum narrow-pass iterations.
    broad_spike_width
        Approximate maximum half-width (in channels) of broad CRs to detect.
        The broad-pass medfilt kernel = ``4 × broad_spike_width + 1``, which
        ensures the CR occupies < 50% of the kernel window even at its widest.
        Set to 0 to disable the broad pass entirely.
        Example: set to 15 for CRs up to 30 channels wide.

    Returns
    -------
    corrected_y
        Same shape as ``y``; unchanged if no spikes detected.
    cosmic_mask
        Bool mask; ``True`` at all corrected channels.
    """
    y1 = _coerce_float_1d_spectrum(y, kernel_size)
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")
    if broad_spike_width < 0:
        raise ValueError("broad_spike_width must be >= 0")

    # Saturated-zero detection (once on original)
    zero_mask = _zero_saturation_mask(y1)
    cumulative_mask = zero_mask.copy()
    current = (
        linear_interpolate_masked_channels_1d(y1, zero_mask) if np.any(zero_mask) else y1.copy()
    )

    # ── Narrow passes (medfilt with kernel_size) ──────────────────────────
    for _ in range(max_passes):
        median_filtered = medfilt(current, kernel_size=kernel_size)
        new_mask, _ = positive_spike_mask_vs_median_smooth(current, median_filtered, threshold)

        if not np.any(new_mask):
            break

        new_mask = _dilate_mask_1d(new_mask)
        cumulative_mask |= new_mask

        if np.all(cumulative_mask):
            warnings.warn(
                "remove_cosmic_rays_1d: all spectral channels flagged — "
                "returning original spectrum unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return y1.copy(), cumulative_mask

        # Per-pass repair keeps current clean for the next narrow pass
        current = linear_interpolate_masked_channels_1d(y1, cumulative_mask)

    # ── Broad pass (large-kernel medfilt on narrow-corrected signal) ──────
    if broad_spike_width > 0:
        # 4×W+1 ensures a CR of half-width W is < 50% of the kernel, so
        # the median remains in the background and the CR is detectable.
        kernel_broad = 4 * broad_spike_width + 1
        # Cap at spectrum length; ensure odd
        kernel_broad = min(kernel_broad, len(current))
        if kernel_broad % 2 == 0:
            kernel_broad -= 1
        if kernel_broad >= 3:
            median_broad = medfilt(current, kernel_size=kernel_broad)
            new_mask_broad, _ = positive_spike_mask_vs_median_smooth(
                current, median_broad, threshold
            )
            # Only add channels not already covered by narrow passes
            new_mask_broad &= ~cumulative_mask

            if np.any(new_mask_broad):
                new_mask_broad = _dilate_mask_1d(new_mask_broad)
                candidate = cumulative_mask | new_mask_broad
                if not np.all(candidate):
                    cumulative_mask = candidate

    # ── SNIP background for repair (clamps repair to ≥ underlying baseline) ──
    n_snip = max(broad_spike_width * 2, _SNIP_REPAIR_ITERATIONS_DEFAULT)
    bg_snip = snip_background_1d(y1, n_snip)

    # ── Final repair ──────────────────────────────────────────────────────
    corrected = repair_masked_channels_1d(y1, cumulative_mask, bg_snip)
    return corrected, cumulative_mask

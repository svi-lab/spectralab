"""1D spectrum cosmic-ray (positive spike) detection and repair."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import medfilt

from ._mad import local_mad_noise, noise_estimate_too_small, robust_mad_noise_with_floor

# Block size (channels) for the local noise estimate used by the narrow and
# broad passes. Small enough to track a shot-noise baseline that ramps by an
# order of magnitude across the spectrum; large enough for a stable MAD
# (needs several dozen points per block).
_LOCAL_NOISE_WINDOW_CHANNELS: int = 200

# Broad-pass kernel multipliers, tried in order after the narrow passes.
# `4*W+1` alone misses a cosmic ray whose width is close to `2*W` (it then
# occupies >=50% of the kernel and dominates the running median instead of
# standing out from it); `8*W+1` catches what the first rung misses without
# requiring the user to size `broad_spike_width` exactly.
_BROAD_KERNEL_MULTIPLIERS: tuple[int, ...] = (4, 8)

# A wide CR's decaying edges taper gradually enough that the last channel or
# two before returning to baseline can sit just under the detection cutoff
# (they're only marginally elevated). A wider dilation than the narrow
# pass's +/-1 sweeps those residual shoulders in too; harmless for real
# features since it only ever grows a broad-pass hit, and the broad pass
# itself already excludes anything narrow enough to be a real PL peak.
_BROAD_DILATE_CHANNELS: int = 3

# A genuinely broad CR sits above the oversized broad-pass median across
# essentially its whole width, so it survives as one long contiguous run.
# A real narrow peak's residual against that same oversized reference
# crosses the threshold only where its cap pokes furthest above the
# (much lower) broad median — a handful of isolated channels, not a
# sustained run. Dropping runs shorter than this fraction of
# broad_spike_width tells the two apart without touching narrow-pass
# detections at all.
_BROAD_MIN_RUN_FRACTION: float = 0.15

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


def _dilate_mask_1d(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Expand a boolean mask by ``n`` channels on each side."""
    out = mask.copy()
    for _ in range(n):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        out = grown
    return out


def _drop_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Clear every contiguous run of ``True`` shorter than ``min_run``."""
    if min_run <= 1 or not np.any(mask):
        return mask
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if run.size < min_run:
            out[run] = False
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
    *,
    noise: np.ndarray | float | None = None,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Mask where positive residual exceeds the MAD gate *and* a relative floor.

    Residual is y − median_smoothed_y. ``noise`` is the per-channel (or
    scalar) noise scale to gate against; when ``None`` (default, and what
    every caller outside this module still gets) it falls back to a single
    scaled-MAD-of-residual for the whole trace — the original behaviour.
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
    if noise is None:
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


def detect_cosmic_mask_1d(
    y: np.ndarray,
    *,
    kernel_size: int = 5,
    threshold: float = 5.0,
    max_passes: int = 3,
    broad_spike_width: int = 15,
) -> np.ndarray:
    """Detect sharp and broad positive spikes in a single 1D spectrum.

    Pure detection — no repair — so callers that need every spectrum's mask
    before repairing any of them (e.g. a cross-spectrum consensus veto) don't
    pay for SNIP background estimation until they know which channels are
    actually being repaired. :func:`remove_cosmic_rays_1d` is this function
    followed by SNIP + repair.

    **Narrow pass** (up to ``max_passes`` iterations): medfilt reference with
    ``kernel_size``.  Catches spikes narrower than ``kernel_size // 2``
    channels.  Uses a *local* noise estimate (blocks of
    ``_LOCAL_NOISE_WINDOW_CHANNELS``) rather than one MAD for the whole
    spectrum, since shot-noise-limited baselines have noise proportional to
    sqrt(intensity) and can span an order of magnitude across one spectrum.

    **Broad pass** (if ``broad_spike_width > 0``): a short kernel ladder —
    medfilt with ``kernel_size = 4 × broad_spike_width + 1``, then
    ``8 × broad_spike_width + 1`` — run on the already narrow-corrected
    signal so narrow spikes do not contaminate the broad reference. The
    second rung catches a CR whose width sits close to the first rung's own
    kernel (where it would occupy ≥50% of the window and bias the median
    instead of standing out from it).

    Parameters
    ----------
    y
        One spectral trace (any numeric dtype; cast to float).
    kernel_size
        Odd ≥ 3; medfilt window for the narrow pass.  Increase for slightly
        wider narrow spikes (e.g. 9–13 for 4–6 channel CRs).
    threshold
        Local-noise multiplier.  Lower → more detections.
    max_passes
        Maximum narrow-pass iterations.
    broad_spike_width
        Approximate maximum half-width (in channels) of broad CRs to detect.
        Set to 0 to disable the broad pass entirely.
        Example: set to 15 for CRs up to 30 channels wide.

    Returns
    -------
    cosmic_mask
        Bool mask, same shape as ``y``; ``True`` at every flagged channel.
    """
    y1 = _coerce_float_1d_spectrum(y, kernel_size)
    if y1.size == 0 or np.all(np.isnan(y1)):
        return np.zeros(y1.shape, dtype=bool)
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

    amplitude_reference = float(np.nanmax(np.abs(y1))) if y1.size else 0.0
    window = min(_LOCAL_NOISE_WINDOW_CHANNELS, y1.size)

    # ── Narrow passes (medfilt with kernel_size) ──────────────────────────
    for _ in range(max_passes):
        median_filtered = medfilt(current, kernel_size=kernel_size)
        residual = current - median_filtered
        noise = local_mad_noise(residual, window, amplitude_reference=amplitude_reference)
        new_mask, _ = positive_spike_mask_vs_median_smooth(
            current, median_filtered, threshold, noise=noise
        )

        if not np.any(new_mask):
            break

        new_mask = _dilate_mask_1d(new_mask)
        cumulative_mask |= new_mask

        if np.all(cumulative_mask):
            warnings.warn(
                "detect_cosmic_mask_1d: all spectral channels flagged — "
                "returning an unflagged mask.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(y1.shape, dtype=bool)

        # Per-pass repair keeps current clean for the next narrow pass
        current = linear_interpolate_masked_channels_1d(y1, cumulative_mask)

    # ── Broad pass(es) (large-kernel medfilt ladder on narrow-corrected signal) ──
    if broad_spike_width > 0:
        min_run = max(3, int(round(_BROAD_MIN_RUN_FRACTION * broad_spike_width)))
        for multiplier in _BROAD_KERNEL_MULTIPLIERS:
            # multiplier×W+1 ensures a CR of half-width W is < 1/multiplier of
            # the kernel, so the median remains in the background and the CR
            # is detectable.
            kernel_broad = multiplier * broad_spike_width + 1
            # Cap at spectrum length; ensure odd
            kernel_broad = min(kernel_broad, len(current))
            if kernel_broad % 2 == 0:
                kernel_broad -= 1
            if kernel_broad < 3:
                continue
            median_broad = medfilt(current, kernel_size=kernel_broad)
            residual_broad = current - median_broad
            noise_broad = local_mad_noise(
                residual_broad, window, amplitude_reference=amplitude_reference
            )
            new_mask_broad, _ = positive_spike_mask_vs_median_smooth(
                current, median_broad, threshold, noise=noise_broad
            )
            # Only add channels not already covered by earlier passes
            new_mask_broad &= ~cumulative_mask
            # A real feature's residual against this oversized reference
            # crosses the cutoff only near its own cap — a short, isolated
            # run. A genuine broad CR sits above the reference across
            # nearly its whole width.
            new_mask_broad = _drop_short_runs(new_mask_broad, min_run)

            if np.any(new_mask_broad):
                new_mask_broad = _dilate_mask_1d(new_mask_broad, _BROAD_DILATE_CHANNELS)
                candidate = cumulative_mask | new_mask_broad
                if not np.all(candidate):
                    cumulative_mask = candidate
                    current = linear_interpolate_masked_channels_1d(y1, cumulative_mask)

    return cumulative_mask


def repair_cosmic_mask_1d(
    y: np.ndarray,
    mask: np.ndarray,
    *,
    broad_spike_width: int = 15,
) -> np.ndarray:
    """Repair channels already flagged in ``mask`` via SNIP-floored interpolation.

    Split out from :func:`remove_cosmic_rays_1d` so a caller that computes
    several spectra's masks together (e.g. a cross-spectrum consensus veto
    that must see every spectrum's mask before repairing any of them) can
    adjust the masks first and repair after, without duplicating the SNIP
    setup. ``broad_spike_width`` only affects the SNIP iteration count, same
    as inside :func:`remove_cosmic_rays_1d`.
    """
    y1 = np.asarray(y, dtype=float)
    n_snip = max(broad_spike_width * 2, _SNIP_REPAIR_ITERATIONS_DEFAULT)
    bg_snip = snip_background_1d(y1, n_snip)
    return repair_masked_channels_1d(y1, mask, bg_snip)


def remove_cosmic_rays_1d(
    y: np.ndarray,
    *,
    kernel_size: int = 5,
    threshold: float = 5.0,
    max_passes: int = 3,
    broad_spike_width: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and repair sharp and broad positive spikes in a single 1D spectrum.

    Detection is :func:`detect_cosmic_mask_1d` (see there for parameter
    semantics). Repair uses the SNIP background as a floor — prevents the
    corrected signal from dipping below the true spectral baseline
    (negative-peak artefact common with pure linear interpolation across
    broad masked regions on sloped features).

    Returns
    -------
    corrected_y
        Same shape as ``y``; unchanged if no spikes detected.
    cosmic_mask
        Bool mask; ``True`` at all corrected channels.
    """
    y1 = _coerce_float_1d_spectrum(y, kernel_size)
    if y1.size == 0 or np.all(np.isnan(y1)):
        return y1.copy(), np.zeros(y1.shape, dtype=bool)

    cumulative_mask = detect_cosmic_mask_1d(
        y1,
        kernel_size=kernel_size,
        threshold=threshold,
        max_passes=max_passes,
        broad_spike_width=broad_spike_width,
    )
    corrected = repair_cosmic_mask_1d(y1, cumulative_mask, broad_spike_width=broad_spike_width)
    return corrected, cumulative_mask

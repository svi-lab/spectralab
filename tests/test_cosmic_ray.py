"""Synthetic cosmic-ray tests: 1D engine, collection engine, map engine.

The 1D engine (UI "1D — per spectrum", default spike_threshold=3.5,
broad_spike_width=15) is designed to catch CRs narrower than ~30 channels
(half-width 15; kernel = 4×15+1 = 61 so the CR is < 50% of the window).

A real PL band is a much broader Gaussian. Those must not be treated as CRs.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cosmic_ray import CosmicRayRemover
from cosmic_ray.mask_1d import remove_cosmic_rays_1d
from tests.factories import gaussian_peak_1d, inject_rect_spike

N = 1024
PL_CENTER = 700
PL_FWHM = 250
PL_AMP = 1000.0
BASELINE = 50.0
CR_START = 40
CR_HEIGHT = 5000.0

# UI / CosmicRayRemover 1D defaults.
_1D = dict(kernel_size=5, threshold=3.5, max_passes=3, broad_spike_width=15)

# Widths the default broad pass is specified to catch (full width ≲ 30).
# 40 channels is already > 50% of the 61-channel kernel, so the median
# sits on the pulse and it is not a CR under the relative-residual gate.
DEFAULT_REMOVABLE = [1, 2, 3, 5, 8, 10, 15, 20, 30]
# Wider than the default kernel can treat as a minority in the window.
BEYOND_DEFAULT = [40, 80, 150]


def _clean_pl() -> np.ndarray:
    return gaussian_peak_1d(N, PL_CENTER, PL_FWHM, PL_AMP, BASELINE)


def _remaining_spike_frac(corrected: np.ndarray, clean: np.ndarray, start: int, width: int) -> float:
    """Peak leftover spike height in the CR window, as a fraction of CR_HEIGHT."""
    region = corrected[start : start + width] - clean[start : start + width]
    return float(np.max(region) / CR_HEIGHT)


# ---------------------------------------------------------------------------
# 1D: isolated CRs of varying width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", DEFAULT_REMOVABLE)
def test_1d_removes_cr_on_baseline_default_params(width):
    clean = _clean_pl()
    y = inject_rect_spike(clean, CR_START, width, CR_HEIGHT)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    assert mask[CR_START : CR_START + width].mean() == pytest.approx(1.0)
    assert _remaining_spike_frac(corr, clean, CR_START, width) < 0.05


@pytest.mark.parametrize("width", BEYOND_DEFAULT)
def test_1d_default_params_cannot_fully_remove_very_wide_pulses(width):
    """A 80–150 channel pulse occupies ≥50% of the default kernel — leftover is expected."""
    clean = _clean_pl()
    y = inject_rect_spike(clean, CR_START, width, CR_HEIGHT)
    corr, _mask = remove_cosmic_rays_1d(y, **_1D)
    assert _remaining_spike_frac(corr, clean, CR_START, width) > 0.5


@pytest.mark.parametrize("width", [50, 80, 100, 150])
def test_1d_scaled_broad_width_removes_wide_cr_on_flat_baseline(width):
    """On a flat baseline (no PL band) a larger kernel can still catch a 150-ch CR."""
    clean = np.full(N, BASELINE, dtype=float)
    y = inject_rect_spike(clean, CR_START, width, CR_HEIGHT)
    bsw = width // 2 + 1
    corr, mask = remove_cosmic_rays_1d(
        y, kernel_size=5, threshold=3.5, max_passes=3, broad_spike_width=bsw
    )
    assert mask[CR_START : CR_START + width].mean() == pytest.approx(1.0)
    assert _remaining_spike_frac(corr, clean, CR_START, width) < 0.05


@pytest.mark.parametrize("width", [1, 3, 5, 10, 15])
def test_1d_removes_narrow_cr_sitting_on_pl_peak(width):
    clean = _clean_pl()
    start = PL_CENTER - width // 2
    y = inject_rect_spike(clean, start, width, CR_HEIGHT)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    assert mask[start : start + width].any()
    leftover = float(np.max(corr[start : start + width] - clean[start : start + width]))
    assert leftover / CR_HEIGHT < 0.05


# ---------------------------------------------------------------------------
# 1D: real PL bands must not be beheaded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fwhm", [80, 150, 250])
def test_1d_does_not_clip_pl_peak_head_without_crs(fwhm):
    """A PL Gaussian (no spikes) must keep its maximum.

    The 1D broad pass currently flags the convex cap of even a 250-channel
    band because residual vs a 61-channel median exceeds a near-zero MAD.
    """
    y = gaussian_peak_1d(N, PL_CENTER, fwhm, PL_AMP, BASELINE)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    drop = (y.max() - corr.max()) / y.max()
    assert not mask[PL_CENTER], f"peak centre flagged, drop={drop:.3f}"
    assert drop < 0.02, f"peak head clipped by {drop:.1%}"


def test_1d_cr_on_baseline_does_not_clip_distant_pl_peak():
    clean = _clean_pl()
    y = inject_rect_spike(clean, CR_START, 5, CR_HEIGHT)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    drop = (clean.max() - corr.max()) / clean.max()
    assert not mask[PL_CENTER]
    assert drop < 0.02


# ---------------------------------------------------------------------------
# Collection engine (≥ 20 spectra): shared PL is the reference, not a CR
# ---------------------------------------------------------------------------


def test_collection_removes_spatial_outlier_cr_and_keeps_shared_pl():
    pl = gaussian_peak_1d(256, 128, 40, PL_AMP, BASELINE)
    stack = np.tile(pl, (25, 1)).astype(float)
    stack[7, 30:33] += CR_HEIGHT
    da = xr.DataArray(stack, dims=("point", "wavelength"))
    out, diag = CosmicRayRemover().remove_cosmic_rays_with_diagnostics(da)

    np.testing.assert_allclose(out.values[0], pl, atol=1e-3)
    leftover = out.values[7, 30:33] - pl[30:33]
    assert np.max(leftover) / CR_HEIGHT < 0.05
    assert not diag["core_mask"][0, 128]


# ---------------------------------------------------------------------------
# Map engine (3-D, ≥ 20 pixels)
# ---------------------------------------------------------------------------


def test_map_engine_flags_only_the_pixel_with_a_cr():
    pl = gaussian_peak_1d(256, 128, 40, PL_AMP, BASELINE)
    cube = np.tile(pl, (8, 8, 1)).astype(float)
    cube[3, 4, 30:33] += CR_HEIGHT
    da = xr.DataArray(cube, dims=("y", "x", "wavelength"))
    out, diag = CosmicRayRemover(force_1d=False).remove_cosmic_rays_with_diagnostics(da)

    assert diag["core_mask"][3, 4, 30:33].all()
    assert not diag["core_mask"][0, 0, 128]
    leftover = out.values[3, 4, 30:33] - pl[30:33]
    assert np.max(leftover) / CR_HEIGHT < 0.05


def test_map_engine_clean_cube_does_not_shift_intensities():
    """Detection is min-subtracted; unflagged pixels must still round-trip."""
    pl = gaussian_peak_1d(256, 128, 80, PL_AMP, BASELINE)
    cube = np.tile(pl, (8, 8, 1)).astype(float)
    da = xr.DataArray(cube, dims=("y", "x", "wavelength"))
    out = CosmicRayRemover(force_1d=False).remove_cosmic_rays(da)
    np.testing.assert_allclose(out.values, cube, atol=1.0)


def test_force_1d_on_a_map_does_not_clip_shared_pl_peak():
    """UI 1D path on a map: a realistic PL band (FWHM 80) must keep its head."""
    pl = gaussian_peak_1d(256, 128, 80, PL_AMP, BASELINE)
    cube = np.tile(pl, (8, 8, 1)).astype(float)
    da = xr.DataArray(cube, dims=("y", "x", "wavelength"))
    out, diag = CosmicRayRemover(force_1d=True).remove_cosmic_rays_with_diagnostics(da)
    drop = (pl.max() - out.values[0, 0].max()) / pl.max()
    assert not diag["cosmic_masks"][0, 0, 128]
    assert drop < 0.02

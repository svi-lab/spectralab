"""Synthetic cosmic-ray tests: 1D engine, collection engine, map engine.

The 1D engine (UI "1D — per spectrum", default spike_threshold=8.0,
broad_spike_width=15) runs a kernel *ladder* after the narrow passes —
medfilt at 4×15+1=61 then 8×15+1=121 — each catching CRs up to just under
half its own kernel width. That extends the removable range to ~60 channels
(vs. ~30 with the single 61-kernel the module used before), at the cost of
also being able to clip a real PL band whose FWHM sits inside that same
extended range when broad_spike_width is left at its small (15) default —
see the FWHM boundary probed in test_1d_does_not_clip_pl_peak_head_without_crs.
In production this is exactly what CosmicRayRemover.broad_width_units (a
physical width resolved per-file from the spectral axis) is for: size the
broad pass to the actual expected CR width, not larger than the narrowest
real feature. Detection also uses a *local* (block-wise) noise estimate
instead of one MAD for the whole spectrum, and a genuinely broad CR must
survive as a long contiguous run — an isolated few-channel excursion (a
real narrow peak poking above an oversized median reference) is dropped
before it's ever added to the mask.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cosmic_ray import CosmicRayRemover, spectral_to_channels
from cosmic_ray._mad import local_mad_noise
from cosmic_ray.mask_1d import remove_cosmic_rays_1d
from tests.factories import (
    gaussian_peak_1d,
    high_res_pl_spectrum,
    inject_plateau_cr,
    inject_rect_spike,
    pl_map_with_cr,
)

N = 1024
PL_CENTER = 700
PL_FWHM = 250
PL_AMP = 1000.0
BASELINE = 50.0
CR_START = 40
CR_HEIGHT = 5000.0

# UI / CosmicRayRemover 1D defaults.
_1D = dict(kernel_size=5, threshold=8.0, max_passes=3, broad_spike_width=15)

# Widths the default broad-pass ladder is specified to catch (full width
# just under half the largest rung, 8×15+1=121, i.e. ≲ 60).
DEFAULT_REMOVABLE = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
# Wider than the largest rung can treat as a minority in the window.
BEYOND_DEFAULT = [80, 120, 150]


def _clean_pl() -> np.ndarray:
    return gaussian_peak_1d(N, PL_CENTER, PL_FWHM, PL_AMP, BASELINE)


def _remaining_spike_frac(
    corrected: np.ndarray, clean: np.ndarray, start: int, width: int
) -> float:
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
    """An 80-150 channel pulse occupies >=50% of even the largest rung — leftover is expected."""
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
        y, kernel_size=5, threshold=8.0, max_passes=3, broad_spike_width=bsw
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


@pytest.mark.parametrize("fwhm", [150, 200, 250])
def test_1d_does_not_clip_pl_peak_head_without_crs(fwhm):
    """A PL Gaussian (no spikes) must keep its maximum.

    fwhm=150 is the practical floor for broad_spike_width=15's ladder: its
    largest rung (8x15+1=121) treats anything narrower as a minority of the
    window and can clip it (see test_1d_clips_narrower_pl_peak_at_undersized_broad_width
    below for the documented failure mode, and CosmicRayRemover.broad_width_units
    for how production avoids it by sizing the broad pass from the real axis).
    """
    y = gaussian_peak_1d(N, PL_CENTER, fwhm, PL_AMP, BASELINE)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    drop = (y.max() - corr.max()) / y.max()
    assert not mask[PL_CENTER], f"peak centre flagged, drop={drop:.3f}"
    assert drop < 0.02, f"peak head clipped by {drop:.1%}"


def test_1d_clips_narrower_pl_peak_at_undersized_broad_width():
    """Documents the trade-off: broad_spike_width must stay well under the
    narrowest real feature's width, or the broad pass's own ladder (up to
    8x broad_spike_width+1) will treat that feature as a broad CR. Here
    broad_spike_width=15 is comparable to an 80-channel-FWHM peak, which
    the default ladder then clips substantially. This is why the UI resolves
    broad_spike_width from a physical width (broad_width_units) using the
    loaded file's own axis spacing, rather than leaving a fixed channel
    count that may be wrong for a given dispersion.
    """
    y = gaussian_peak_1d(N, PL_CENTER, 80, PL_AMP, BASELINE)
    corr, mask = remove_cosmic_rays_1d(y, **_1D)
    drop = (y.max() - corr.max()) / y.max()
    assert mask[PL_CENTER]
    assert drop > 0.1


def test_1d_narrow_broad_spike_width_keeps_wider_pl_peak():
    """The same 80-channel-FWHM peak survives once broad_spike_width is
    sized *well under* the real feature's width — the ladder's kernels
    (4x/8x broad_spike_width+1) then stay small enough that, centred on the
    peak, they still see mostly-elevated signal and the median tracks the
    peak instead of its wings. This is the direction physical-unit
    resolution (broad_width_units) pushes production configuration: size
    the broad pass to the actual expected CR, not to "be safe" and oversize
    it — an oversized kernel's median falls into the peak's own wings and
    *increases* the apparent residual instead of avoiding it (see the
    bsw=68 vs bsw=8 contrast one would find by sweeping this)."""
    y = gaussian_peak_1d(N, PL_CENTER, 80, PL_AMP, BASELINE)
    corr, mask = remove_cosmic_rays_1d(
        y, kernel_size=5, threshold=8.0, max_passes=3, broad_spike_width=8
    )
    drop = (y.max() - corr.max()) / y.max()
    assert not mask[PL_CENTER]
    assert drop < 0.02


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


# ---------------------------------------------------------------------------
# High-dispersion instrument regression (the file that exposed this bug):
# ~9300 channels over 360-700 nm, a narrow real line, a broader real
# shoulder, shot noise, and — separately — a wide plateau-shaped CR.
# ---------------------------------------------------------------------------


def _resolved_widths(channel_width: float, narrow_nm: float = 0.2, broad_nm: float = 2.5):
    spike_width = spectral_to_channels(narrow_nm, channel_width, minimum=3, odd=True)
    broad_spike_width = spectral_to_channels(broad_nm, channel_width, minimum=1, odd=False)
    return spike_width, broad_spike_width


def test_1d_keeps_narrow_real_line_at_default_params():
    """The ~5sigma, ~11-channel real line must survive untouched.

    This is the false positive that motivated the fix: at the old UI
    defaults (spike_width=3 channels, threshold=0.10 — the degenerate
    regime where the noise gate is disabled, see mask_1d.py's
    _MIN_RELATIVE_RESIDUAL) this line lost its cap.
    """
    x, clean, noisy = high_res_pl_spectrum(seed=0)
    channel_width = float(np.median(np.diff(x)))
    spike_width, broad_spike_width = _resolved_widths(channel_width)
    corr, mask = remove_cosmic_rays_1d(
        noisy,
        kernel_size=spike_width,
        threshold=8.0,
        max_passes=3,
        broad_spike_width=broad_spike_width,
    )
    line = np.where((x >= 361.4) & (x <= 362.4))[0]
    assert not mask[line].any()
    assert corr[line].max() == pytest.approx(noisy[line].max())


def test_1d_removes_wide_plateau_cr():
    """A 58-channel plateau CR (the real file's actual shape at 519-521 nm)
    must be substantially removed, without disturbing the real features."""
    x, clean, noisy = high_res_pl_spectrum(seed=0)
    channel_width = float(np.median(np.diff(x)))
    spike_width, broad_spike_width = _resolved_widths(channel_width)
    cr_idx = int(np.argmin(np.abs(x - 550.0)))
    y = inject_plateau_cr(noisy, cr_idx, 58, 3000.0)
    corr, mask = remove_cosmic_rays_1d(
        y, kernel_size=spike_width, threshold=8.0, max_passes=3, broad_spike_width=broad_spike_width
    )
    cr_region = slice(cr_idx, cr_idx + 58)
    assert mask[cr_region].mean() > 0.8
    leftover = (corr[cr_region] - clean[cr_region]).max()
    assert leftover < 0.3 * 3000.0

    line = np.where((x >= 361.4) & (x <= 362.4))[0]
    shoulder = np.where((x >= 368.5) & (x <= 370.5))[0]
    assert not mask[line].any()
    assert not mask[shoulder].any()


def test_1d_false_positive_rate_on_shot_noise_tail():
    """Across the whole spectrum (shot noise ramping baseline, two real
    features, no injected CR), only a handful of channels — not hundreds —
    should ever be flagged. Pre-fix, the degenerate old defaults flagged
    36,364 channels across a 457-pixel map of this same instrument."""
    x, clean, noisy = high_res_pl_spectrum(seed=0)
    channel_width = float(np.median(np.diff(x)))
    spike_width, broad_spike_width = _resolved_widths(channel_width)
    _corr, mask = remove_cosmic_rays_1d(
        noisy,
        kernel_size=spike_width,
        threshold=8.0,
        max_passes=3,
        broad_spike_width=broad_spike_width,
    )
    assert mask.sum() < 30


def test_local_noise_tracks_intensity():
    """local_mad_noise must report more noise where the underlying spread
    actually is larger — a single whole-spectrum MAD cannot, and that
    mismatch is what let shot noise in a bright region masquerade as CRs."""
    rng = np.random.default_rng(0)
    n = 2000
    residual = np.concatenate([rng.normal(0.0, 2.0, n // 2), rng.normal(0.0, 20.0, n // 2)])
    noise = local_mad_noise(residual, window=200, amplitude_reference=100.0)
    dim_region = noise[: n // 2 - 100]
    bright_region = noise[n // 2 + 100 :]
    assert bright_region.mean() > 5 * dim_region.mean()
    # Rough sanity check against the true sigma of each half (MAD -> sigma).
    assert dim_region.mean() == pytest.approx(2.0, rel=0.4)
    assert bright_region.mean() == pytest.approx(20.0, rel=0.4)


def test_consensus_veto_keeps_shared_line_and_removes_single_pixel_cr():
    """Loop-1D engine on a map: a line present in every pixel must never be
    touched by the consensus veto; a CR present in exactly one pixel must
    still be caught (the veto only suppresses near-universal detections)."""
    cube = pl_map_with_cr(n_row=8, n_col=8, n_spec=512, cr_row=3, cr_col=4, seed=0)
    da = xr.DataArray(cube, dims=("row", "col", "wavelength"))
    crr = CosmicRayRemover(
        spike_width=5,
        spike_threshold=8.0,
        spike_passes=3,
        broad_spike_width=15,
        force_1d=True,
        consensus_veto_fraction=0.3,
    )
    out, diag = crr.remove_cosmic_rays_with_diagnostics(da)
    masks = diag["cosmic_masks"]

    line_center = 512 // 3
    line = slice(line_center - 5, line_center + 5)
    assert not masks[:, :, line].any(), "shared line was flagged/vetoed incorrectly"
    for r in range(cube.shape[0]):
        for c in range(cube.shape[1]):
            assert out.values[r, c, line].max() == pytest.approx(cube[r, c, line].max())

    cr_start = 512 * 2 // 3
    cr_region = slice(cr_start, cr_start + 40)
    assert masks[3, 4, cr_region].any(), "single-pixel CR should still be detected"
    assert out.values[3, 4].max() < cube[3, 4].max()


# ---------------------------------------------------------------------------
# Physical-unit width conversion (backend/cosmic_ray/_units.py)
# ---------------------------------------------------------------------------


def test_spectral_width_to_channels_roundtrip():
    """2.5 nm broad width resolves to ~68 channels at the real file's
    dispersion (0.037 nm/channel) and to a much smaller count on a coarser
    grating — always odd-when-requested and >= the given minimum."""
    fine = spectral_to_channels(2.5, 0.037, minimum=1, odd=False)
    coarse = spectral_to_channels(2.5, 0.5, minimum=1, odd=False)
    assert fine == pytest.approx(68, abs=2)
    assert coarse == pytest.approx(5, abs=1)
    assert fine > coarse

    odd_width = spectral_to_channels(0.2, 0.037, minimum=3, odd=True)
    assert odd_width % 2 == 1
    assert odd_width >= 3

    # span<=0 always means "disabled", regardless of minimum.
    assert spectral_to_channels(0.0, 0.037, minimum=5, odd=False) == 0
    assert spectral_to_channels(-1.0, 0.037, minimum=5, odd=False) == 0

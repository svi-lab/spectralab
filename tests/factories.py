"""Synthetic SpectralDataset helpers for pipeline / cache tests."""

from __future__ import annotations

import numpy as np
import xarray as xr

from backend._shared.dataset import SpectralDataset


def make_map(
    values: np.ndarray,
    *,
    spectral_units: str = "Nanometer",
    spectral_unit: str = "nm",
    laser_nm: float | None = 355.0,
) -> SpectralDataset:
    """Build a SpectralDataset from a ``(n_row, n_col, n_spectral)`` array."""
    if values.ndim != 3:
        raise ValueError(f"expected 3-D map, got shape {values.shape}")
    n_row, n_col, n_spec = values.shape
    spectral_dim = "wavelength"
    da = xr.DataArray(
        values,
        dims=("row", "column", spectral_dim),
        coords={
            "row": np.arange(n_row),
            "column": np.arange(n_col),
            spectral_dim: np.linspace(400.0, 700.0, n_spec),
        },
        attrs={"spectral_units": spectral_units},
    )
    da.coords[spectral_dim].attrs["units"] = spectral_unit
    spec = da.coords[spectral_dim].values
    return SpectralDataset(
        da=da,
        spectral_dim=spectral_dim,
        spectral_units=spectral_units,
        spectral_unit=spectral_unit,
        spec_min=float(spec.min()),
        spec_max=float(spec.max()),
        laser_nm=laser_nm,
        is_map=True,
        image_arr=None,
        image_meta=None,
        laser_power=float("nan"),
        exposure_time=float("nan"),
        comment="",
        dims=da.dims,
        shape=da.shape,
        ndim=da.ndim,
        is_valid=True,
        validation_msg="",
    )


def gaussian_map(
    n_row: int = 4,
    n_col: int = 5,
    n_spec: int = 48,
    dtype=np.float64,
) -> np.ndarray:
    """Smooth PL-like cube: a Gaussian peak plus a small offset, no spikes."""
    x = np.linspace(-3.0, 3.0, n_spec)
    peak = np.exp(-0.5 * x**2)
    cube = np.empty((n_row, n_col, n_spec), dtype=dtype)
    for r in range(n_row):
        for c in range(n_col):
            amp = 50.0 + 5.0 * r + c
            cube[r, c] = amp * peak + 2.0
    return cube


def gaussian_peak_1d(
    n: int,
    center: int,
    fwhm: float,
    amp: float = 1000.0,
    baseline: float = 50.0,
) -> np.ndarray:
    """Gaussian in channel space. ``fwhm`` is full width at half-maximum in channels."""
    x = np.arange(n, dtype=float)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma = max(sigma, 1e-9)
    return baseline + amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def inject_rect_spike(
    y: np.ndarray,
    start: int,
    width: int,
    height: float,
) -> np.ndarray:
    """Copy of ``y`` with a rectangular cosmic-ray-like pulse added."""
    out = np.asarray(y, dtype=float).copy()
    out[start : start + width] += height
    return out


def inject_plateau_cr(
    y: np.ndarray,
    start: int,
    width: int,
    height: float,
    *,
    edge: int = 4,
) -> np.ndarray:
    """Copy of ``y`` with a flat-topped pulse ramping up/down over ``edge``
    channels at each side, instead of a hard rectangular step.

    Matches the shape actually observed on a real cosmic ray (a sharp rise,
    a plateau, then a decaying tail) more closely than
    :func:`inject_rect_spike`'s pure step — the tapered edges are what
    exposed the repair-quality gap a plain ±1-channel mask dilation left
    behind (see `_BROAD_DILATE_CHANNELS` in mask_1d.py).
    """
    out = np.asarray(y, dtype=float).copy()
    edge = max(int(edge), 1)
    n = out.size
    ramp = np.linspace(0.0, 1.0, edge, endpoint=False)
    profile = np.concatenate([ramp, np.ones(max(width - 2 * edge, 0)), ramp[::-1]])
    lo = max(start, 0)
    hi = min(start + profile.size, n)
    out[lo:hi] += height * profile[: hi - lo]
    return out


def high_res_pl_spectrum(
    n: int = 9341,
    lo: float = 360.0,
    hi: float = 700.0,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic high-dispersion PL spectrum reproducing the instrument that
    exposed the CRR false-positive bug: ~9300 channels over 360-700 nm
    (~0.037 nm/channel), a baseline ramping ~80 -> 500 counts, a narrow real
    emission line (~0.4 nm FWHM, ~11 channels), a broader real shoulder
    (~1.5 nm FWHM, ~40 channels), and Poisson-like (shot) noise so sigma
    grows with sqrt(intensity) — the property a single whole-spectrum MAD
    estimate gets wrong (see `local_mad_noise` in `cosmic_ray/_mad.py`).

    Returns ``(wavelength, clean, noisy)``.
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(lo, hi, n)
    channel_width = (hi - lo) / (n - 1)
    baseline = np.linspace(80.0, 500.0, n)
    line = gaussian_peak_1d(
        n, (361.9 - lo) / channel_width, fwhm=0.4 / channel_width, amp=80.0, baseline=0.0
    )
    shoulder = gaussian_peak_1d(
        n, (369.5 - lo) / channel_width, fwhm=1.5 / channel_width, amp=70.0, baseline=0.0
    )
    clean = baseline + line + shoulder
    noisy = clean + rng.normal(0.0, np.sqrt(np.maximum(clean, 1.0)))
    return wavelength, clean, noisy


def pl_map_with_cr(
    n_row: int = 8,
    n_col: int = 8,
    n_spec: int = 512,
    *,
    cr_row: int = 3,
    cr_col: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """``(n_row, n_col, n_spec)`` cube: a real narrow emission line present
    in every pixel (the shared-feature case a consensus veto must protect),
    plus one plateau-shaped cosmic ray in exactly one pixel (the case a
    consensus veto must leave alone). Per-pixel amplitude and noise vary
    slightly, like real data.
    """
    rng = np.random.default_rng(seed)
    line_center = n_spec // 3
    cube = np.empty((n_row, n_col, n_spec), dtype=float)
    for r in range(n_row):
        for c in range(n_col):
            amp = 80.0 + 5.0 * rng.standard_normal()
            spectrum = gaussian_peak_1d(n_spec, line_center, fwhm=8.0, amp=amp, baseline=100.0)
            spectrum = spectrum + rng.normal(0.0, 2.0, n_spec)
            cube[r, c] = spectrum
    cube[cr_row, cr_col] = inject_plateau_cr(cube[cr_row, cr_col], n_spec * 2 // 3, 40, 3000.0)
    return cube


def spectrum_da(
    values: np.ndarray,
    *,
    laser_nm: float | None = 355.0,
    wavelength: np.ndarray | None = None,
    units: str = "nm",
) -> xr.DataArray:
    """1-D spectrum DataArray with a wavelength-style spectral axis."""
    values = np.asarray(values, dtype=float)
    if wavelength is None:
        wavelength = np.arange(values.size, dtype=float)
    da = xr.DataArray(
        values,
        dims=("wavelength",),
        coords={"wavelength": wavelength},
        attrs={"Filename": "synth"},
    )
    da.coords["wavelength"].attrs["units"] = units
    if laser_nm is not None:
        da.attrs["laser_wavelength_nm"] = laser_nm
    return da

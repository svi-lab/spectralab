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

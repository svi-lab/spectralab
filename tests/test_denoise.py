"""PCA and per-spectrum smoother denoising on synthetic stacks."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra_cleaner import Denoiser
from spectra_smoother import SpectraSmoother
from tests.factories import gaussian_peak_1d, spectrum_da


def _shared_peak_stack(
    n_spectra: int = 40,
    n_channels: int = 256,
    noise: float = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-4.0, 4.0, n_channels)
    shape = np.exp(-0.5 * x**2)
    amps = 80.0 + 20.0 * rng.random(n_spectra)
    clean = amps[:, None] * shape[None, :] + 5.0
    noisy = clean + rng.normal(0.0, noise, clean.shape)
    return clean, noisy


def test_pca_reduces_uncorrelated_noise_and_keeps_shared_shape():
    clean, noisy = _shared_peak_stack()
    da = xr.DataArray(noisy, dims=("point", "wavelength"))
    out = Denoiser(n_components=1, subtract_min=True, restore_min=True).denoise(da)
    rmse_in = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rmse_out = float(np.sqrt(np.mean((out.values - clean) ** 2)))
    assert rmse_out < 0.5 * rmse_in
    # Shared Gaussian centre should not jump by more than a couple of channels.
    assert abs(int(np.argmax(out.values[0])) - int(np.argmax(clean[0]))) <= 3


def test_pca_preserves_all_nan_rows():
    clean, noisy = _shared_peak_stack(n_spectra=12, n_channels=64, noise=4.0)
    noisy[3] = np.nan
    da = xr.DataArray(noisy, dims=("point", "wavelength"))
    out = Denoiser(n_components=1, subtract_min=True).denoise(da)
    assert np.isnan(out.values[3]).all()
    assert np.isfinite(out.values[0]).all()


def test_pca_rejects_degenerate_one_row_stack():
    """A 1-D array is routed to the smoother; a 2-D stack of one spectrum is not."""
    y = gaussian_peak_1d(128, 64, 40)
    da_1d = spectrum_da(y, laser_nm=None)
    out = Denoiser(n_components=1).denoise(da_1d)
    assert out.shape == da_1d.shape

    da_row = xr.DataArray(y[None, :], dims=("point", "wavelength"))
    with pytest.raises(ValueError, match="more than one spectrum"):
        Denoiser(n_components=1).denoise(da_row)


def test_savgol_smoother_reduces_noise_without_moving_the_peak():
    rng = np.random.default_rng(2)
    clean = gaussian_peak_1d(256, 128, 40, amp=800.0, baseline=20.0)
    noisy = clean + rng.normal(0.0, 15.0, clean.shape)
    da = xr.DataArray(noisy, dims=("wavelength",))
    out = SpectraSmoother(method="savgol", window_length=11, polyorder=3).smooth(da)
    rmse_in = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rmse_out = float(np.sqrt(np.mean((out.values - clean) ** 2)))
    assert rmse_out < rmse_in
    assert abs(int(np.argmax(out.values)) - 128) <= 2


def test_wavelet_smoother_keeps_a_sharp_peak_location():
    pytest.importorskip("pywt")
    rng = np.random.default_rng(3)
    clean = gaussian_peak_1d(256, 128, 20, amp=800.0, baseline=20.0)
    noisy = clean + rng.normal(0.0, 12.0, clean.shape)
    da = xr.DataArray(noisy, dims=("wavelength",))
    out = SpectraSmoother(method="wavelet").smooth(da)
    assert abs(int(np.argmax(out.values)) - 128) <= 3
    assert float(out.values.max()) > 0.6 * clean.max()


def test_denoiser_per_spectrum_uses_smoother_on_a_stack():
    clean, noisy = _shared_peak_stack(n_spectra=8, n_channels=128, noise=10.0)
    da = xr.DataArray(noisy, dims=("point", "wavelength"))
    out = Denoiser(per_spectrum=True).denoise(da)
    assert out.shape == da.shape
    rmse_in = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rmse_out = float(np.sqrt(np.mean((out.values - clean) ** 2)))
    assert rmse_out < rmse_in

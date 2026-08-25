"""Peak fitter batch path, warm-start, and parallelism."""

from __future__ import annotations

import numpy as np

from backend.peak_fitter import BandSpec, PeakFitter, fit_map_gaussian
from tests.factories import gaussian_map, make_map


def _single_band(center: float = 550.0) -> list[BandSpec]:
    return [
        BandSpec(
            center_guess=center,
            center_min=center - 40.0,
            center_max=center + 40.0,
            label="P1",
        )
    ]


def test_fit_curves_false_skips_component_curves():
    x = np.linspace(400.0, 700.0, 64)
    y = 100.0 * np.exp(-0.5 * ((x - 550.0) / 20.0) ** 2) + 5.0
    result = PeakFitter().fit(x, y, _single_band(), curves=False)
    assert result.success
    assert result.bands[0].curve.size == 0


def test_batch_sequential_matches_parallel():
    da = make_map(gaussian_map(n_row=3, n_col=4, n_spec=64)).da
    bands = _single_band()

    seq = fit_map_gaussian(da, bands, n_jobs=1, max_nfev=500)
    par = fit_map_gaussian(da, bands, n_jobs=2, max_nfev=500)

    np.testing.assert_allclose(seq.r_squared_map, par.r_squared_map, equal_nan=True)
    np.testing.assert_allclose(
        seq.band_results["P1"]["center"], par.band_results["P1"]["center"], equal_nan=True
    )
    assert seq.n_fitted == par.n_fitted


def test_batch_warm_start_survives_nan_gap():
    values = gaussian_map(n_row=1, n_col=3, n_spec=64)
    values[0, 1, :] = np.nan
    da = make_map(values).da
    bands = _single_band()

    result = fit_map_gaussian(da, bands, n_jobs=1, warm_start=True, max_nfev=500)

    assert result.n_fitted == 2
    assert result.n_skipped_nan == 1
    assert np.isfinite(result.band_results["P1"]["center"][0, 0])
    assert np.isnan(result.band_results["P1"]["center"][0, 1])
    assert np.isfinite(result.band_results["P1"]["center"][0, 2])


def test_batch_all_nan_map():
    values = np.full((2, 2, 32), np.nan)
    da = make_map(values).da
    result = fit_map_gaussian(da, _single_band(), n_jobs=1)
    assert result.n_fitted == 0
    assert result.n_skipped_nan == 4

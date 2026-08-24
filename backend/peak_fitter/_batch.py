"""Per-pixel Gaussian deconvolution across an entire map: :func:`fit_map_gaussian`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ._fitter import BandSpec, PeakFitter

_PARAM_NAMES = ("center", "amplitude", "sigma", "fwhm", "area")


@dataclass
class BatchFitResult:
    """Per-pixel fit results reshaped to the map's spatial shape.

    ``band_results[label][param]`` is a ``(n_row, n_col)`` ndarray, directly
    consumable by ``frontend.map_chart.make_scalar_map_fig``.
    """

    band_results: dict[str, dict[str, np.ndarray]]
    r_squared_map: np.ndarray
    reduced_chi_square_map: np.ndarray
    success_map: np.ndarray
    n_fitted: int
    n_skipped_nan: int
    n_failed: int


def fit_map_gaussian(
    da_map: xr.DataArray,
    bands: list[BandSpec],
    *,
    spectral_dim: str | None = None,
    warm_start: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchFitResult:
    """Fit every pixel of a 3D map independently with the same band
    specification, warm-starting each pixel from the previous pixel's
    converged parameters for speed/stability.

    Row-major traversal order (row outer, column inner), matching the
    ``reshape(-1, n_spectral)`` convention used elsewhere in this codebase.
    NaN (dead/oversaturated) pixels are skipped entirely and left NaN in
    every output array. A pixel whose fit raises is recorded in
    ``n_failed`` and also left NaN, and breaks the warm-start chain (the
    next pixel falls back to the ``bands`` guesses) so one bad pixel can't
    poison its neighbors.
    """
    if da_map.ndim != 3:
        raise ValueError(f"fit_map_gaussian needs a 3D map DataArray; got ndim={da_map.ndim}")

    sdim = spectral_dim or da_map.dims[-1]
    spatial_dims = [d for d in da_map.dims if d != sdim]
    if len(spatial_dims) != 2:
        raise ValueError(
            "fit_map_gaussian expects exactly 2 spatial dims besides "
            f"{sdim!r}; got dims={da_map.dims!r}"
        )

    da_w = da_map.transpose(*spatial_dims, sdim)
    n_row, n_col, _ = da_w.shape
    x = da_w.coords[sdim].values.astype(float)
    values = da_w.values

    labels = [b.label or f"Band {i + 1}" for i, b in enumerate(bands)]
    band_results: dict[str, dict[str, np.ndarray]] = {
        label: {p: np.full((n_row, n_col), np.nan) for p in _PARAM_NAMES} for label in labels
    }
    r_squared_map = np.full((n_row, n_col), np.nan)
    reduced_chi_square_map = np.full((n_row, n_col), np.nan)
    success_map = np.zeros((n_row, n_col), dtype=bool)

    fitter = PeakFitter()
    n_fitted = n_skipped_nan = n_failed = 0
    prev_params = None
    total = n_row * n_col
    done = 0

    for r in range(n_row):
        for c in range(n_col):
            done += 1
            y = values[r, c, :]
            if np.all(np.isnan(y)):
                n_skipped_nan += 1
                prev_params = None
            else:
                try:
                    result = fitter.fit(
                        x,
                        y,
                        bands,
                        params_init=prev_params if warm_start else None,
                    )
                except (ValueError, RuntimeError):
                    n_failed += 1
                    prev_params = None
                else:
                    r_squared_map[r, c] = result.r_squared
                    reduced_chi_square_map[r, c] = result.reduced_chi_square
                    success_map[r, c] = result.success
                    for label, band_result in zip(labels, result.bands):
                        band_results[label]["center"][r, c] = band_result.center
                        band_results[label]["amplitude"][r, c] = band_result.amplitude
                        band_results[label]["sigma"][r, c] = band_result.sigma
                        band_results[label]["fwhm"][r, c] = band_result.fwhm
                        band_results[label]["area"][r, c] = band_result.area
                    n_fitted += 1
                    prev_params = (
                        result.raw_lmfit_result.params if warm_start and result.success else None
                    )

            if progress_callback is not None:
                progress_callback(done, total)

    return BatchFitResult(
        band_results=band_results,
        r_squared_map=r_squared_map,
        reduced_chi_square_map=reduced_chi_square_map,
        success_map=success_map,
        n_fitted=n_fitted,
        n_skipped_nan=n_skipped_nan,
        n_failed=n_failed,
    )


__all__ = ["fit_map_gaussian", "BatchFitResult"]

"""Per-pixel Gaussian deconvolution across an entire map: :func:`fit_map_gaussian`."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from ._fitter import BandSpec, PeakFitter, _build_composite_model

_PARAM_NAMES = ("center", "amplitude", "sigma", "fwhm", "area")
_BATCH_MAX_NFEV = 500


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


@dataclass
class _RowFitResult:
    band_results: dict[str, dict[str, np.ndarray]]
    r_squared: np.ndarray
    reduced_chi_square: np.ndarray
    success: np.ndarray
    n_fitted: int
    n_skipped_nan: int
    n_failed: int


def _fit_map_row(
    r: int,
    x: np.ndarray,
    values: np.ndarray,
    n_col: int,
    bands: list[BandSpec],
    labels: list[str],
    model,
    prefixes: list[str],
    *,
    warm_start: bool,
    max_nfev: int,
) -> _RowFitResult:
    """Fit one map row; warm-start propagates along columns within the row."""
    fitter = PeakFitter()
    last_success = None
    band_results: dict[str, dict[str, np.ndarray]] = {
        label: {p: np.full(n_col, np.nan) for p in _PARAM_NAMES} for label in labels
    }
    r_squared = np.full(n_col, np.nan)
    reduced_chi_square = np.full(n_col, np.nan)
    success = np.zeros(n_col, dtype=bool)
    n_fitted = n_skipped_nan = n_failed = 0

    for c in range(n_col):
        y = values[r, c, :]
        if np.all(np.isnan(y)):
            n_skipped_nan += 1
            continue
        try:
            result = fitter.fit(
                x,
                y,
                bands,
                params_init=last_success if warm_start else None,
                curves=False,
                max_nfev=max_nfev,
                model=model,
                prefixes=prefixes,
            )
        except (ValueError, RuntimeError):
            n_failed += 1
            continue

        r_squared[c] = result.r_squared
        reduced_chi_square[c] = result.reduced_chi_square
        success[c] = result.success
        for label, band_result in zip(labels, result.bands):
            band_results[label]["center"][c] = band_result.center
            band_results[label]["amplitude"][c] = band_result.amplitude
            band_results[label]["sigma"][c] = band_result.sigma
            band_results[label]["fwhm"][c] = band_result.fwhm
            band_results[label]["area"][c] = band_result.area
        n_fitted += 1
        if warm_start and result.success:
            last_success = result.raw_lmfit_result.params

    return _RowFitResult(
        band_results=band_results,
        r_squared=r_squared,
        reduced_chi_square=reduced_chi_square,
        success=success,
        n_fitted=n_fitted,
        n_skipped_nan=n_skipped_nan,
        n_failed=n_failed,
    )


def _default_n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def fit_map_gaussian(
    da_map: xr.DataArray,
    bands: list[BandSpec],
    *,
    spectral_dim: str | None = None,
    warm_start: bool = True,
    max_nfev: int = _BATCH_MAX_NFEV,
    n_jobs: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchFitResult:
    """Fit every pixel of a 3D map independently with the same band
    specification, warm-starting each pixel from the previous pixel's
    converged parameters for speed/stability.

    Row-major traversal order (row outer, column inner), matching the
    ``reshape(-1, n_spectral)`` convention used elsewhere in this codebase.
    NaN (dead/oversaturated) pixels are skipped entirely and left NaN in
    every output array. A pixel whose fit raises is recorded in
    ``n_failed`` and also left NaN; warm-start keeps the last *successful*
    parameters across NaN gaps so one bad pixel can't poison its neighbors.

    Rows are processed in parallel when ``n_jobs > 1`` (default: CPU count − 1).
    Warm-start chains stay within each row; rows are independent.
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

    # Seed the composite model from the first finite spectrum.
    seed_y = None
    for r in range(n_row):
        for c in range(n_col):
            y = values[r, c, :]
            if not np.all(np.isnan(y)):
                seed_y = y
                break
        if seed_y is not None:
            break
    if seed_y is None:
        return BatchFitResult(
            band_results=band_results,
            r_squared_map=r_squared_map,
            reduced_chi_square_map=reduced_chi_square_map,
            success_map=success_map,
            n_fitted=0,
            n_skipped_nan=n_row * n_col,
            n_failed=0,
        )

    finite = np.isfinite(x) & np.isfinite(seed_y)
    x_fit = x[finite]
    y_seed = seed_y[finite]
    model, _, prefixes = _build_composite_model(x_fit, y_seed, bands)

    jobs = n_jobs if n_jobs is not None else _default_n_jobs()
    total = n_row * n_col
    n_fitted = n_skipped_nan = n_failed = 0

    row_kwargs = dict(
        x=x,
        values=values,
        n_col=n_col,
        bands=bands,
        labels=labels,
        model=model,
        prefixes=prefixes,
        warm_start=warm_start,
        max_nfev=max_nfev,
    )

    if jobs <= 1:
        for r in range(n_row):
            row_out = _fit_map_row(r, **row_kwargs)
            _merge_row(
                r,
                row_out,
                band_results,
                r_squared_map,
                reduced_chi_square_map,
                success_map,
            )
            n_fitted += row_out.n_fitted
            n_skipped_nan += row_out.n_skipped_nan
            n_failed += row_out.n_failed
            if progress_callback is not None:
                progress_callback((r + 1) * n_col, total)
    else:
        parallel = Parallel(n_jobs=jobs, backend="threading", return_as="generator")
        for r, row_out in enumerate(
            parallel(delayed(_fit_map_row)(r, **row_kwargs) for r in range(n_row))
        ):
            _merge_row(
                r,
                row_out,
                band_results,
                r_squared_map,
                reduced_chi_square_map,
                success_map,
            )
            n_fitted += row_out.n_fitted
            n_skipped_nan += row_out.n_skipped_nan
            n_failed += row_out.n_failed
            if progress_callback is not None:
                progress_callback((r + 1) * n_col, total)

    return BatchFitResult(
        band_results=band_results,
        r_squared_map=r_squared_map,
        reduced_chi_square_map=reduced_chi_square_map,
        success_map=success_map,
        n_fitted=n_fitted,
        n_skipped_nan=n_skipped_nan,
        n_failed=n_failed,
    )


def _merge_row(
    r: int,
    row_out: _RowFitResult,
    band_results: dict[str, dict[str, np.ndarray]],
    r_squared_map: np.ndarray,
    reduced_chi_square_map: np.ndarray,
    success_map: np.ndarray,
) -> None:
    for label in band_results:
        for param in _PARAM_NAMES:
            band_results[label][param][r, :] = row_out.band_results[label][param]
    r_squared_map[r, :] = row_out.r_squared
    reduced_chi_square_map[r, :] = row_out.reduced_chi_square
    success_map[r, :] = row_out.success


__all__ = ["fit_map_gaussian", "BatchFitResult", "_BATCH_MAX_NFEV"]

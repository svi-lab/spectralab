# -*- coding: utf-8 -*-
"""Per-spectrum normalization — area and min-max methods."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)

NORM_METHODS = ("min_max", "area")


def _trapz_y(y: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _normalize_numpy_block(
    spectra_2d: np.ndarray,
    method: str,
    x_values: np.ndarray,
) -> np.ndarray:
    """Normalize a 2D ``(n_spectra, n_points)`` array in-place-safe."""
    if method == "area":
        denom = _trapz_y(spectra_2d, x_values, axis=-1)[:, np.newaxis]
        denom = np.where(np.abs(denom) < np.finfo(float).eps, 1.0, denom)
        out = spectra_2d / denom
    elif method == "min_max":
        s_min = spectra_2d.min(axis=1, keepdims=True)
        s_max = spectra_2d.max(axis=1, keepdims=True)
        rng = s_max - s_min
        rng = np.where(rng < np.finfo(float).eps, 1.0, rng)
        out = (spectra_2d - s_min) / rng
    else:
        raise ValueError(
            f"normalize method {method!r} is not recognised. "
            f"Must be one of {NORM_METHODS!r}."
        )
    # Shift each spectrum so its minimum is exactly 0.
    # For min_max this is a no-op; for area it removes a constant baseline offset.
    out = out - out.min(axis=-1, keepdims=True)
    return out


def _make_apply_ufunc_kernel(method: str, x_values: np.ndarray):
    def _kernel(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, orig_shape[-1])
        return _normalize_numpy_block(arr_2d, method, x_values).reshape(orig_shape)
    return _kernel


def normalize(
    input_spectra: xr.DataArray | np.ndarray,
    method: str = "area",
    *,
    spectral_dim: str | None = None,
) -> xr.DataArray | np.ndarray:
    """Scale spectra along the spectral axis.

    Parameters
    ----------
    input_spectra
        DataArray or 2D ndarray of shape ``(n_spectra, n_points)``.
    method
        ``"area"`` — divide each spectrum by its trapezoidal integral.
        ``"min_max"`` — scale each spectrum to the [0, 1] range.
    spectral_dim
        Spectral dimension name when ``input_spectra`` is a DataArray
        (default: last dimension).

    Returns
    -------
    Same type as input; DataArrays carry updated ``attrs["treatments"]``.
    Dask-backed DataArrays are processed chunk-by-chunk without loading
    all data into RAM.
    """
    if method not in NORM_METHODS:
        raise ValueError(
            f"normalize method {method!r} is not recognised. "
            f"Must be one of {NORM_METHODS!r}."
        )

    if isinstance(input_spectra, xr.DataArray):
        return _normalize_dataarray(input_spectra, method, spectral_dim)

    spectra = np.asarray(input_spectra, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("ndarray input must be 2D with shape (n_spectra, n_points)")
    x_values = np.arange(spectra.shape[-1], dtype=float)
    return _normalize_numpy_block(spectra, method, x_values)


def _normalize_dataarray(
    da: xr.DataArray,
    method: str,
    spectral_dim: str | None,
) -> xr.DataArray:
    sdim = resolve_spectral_dim(da, spectral_dim)
    da_w, orig_order = transpose_spectral_last(da, sdim)
    x_values = da_w[sdim].values

    if da_w.chunks is not None:
        kernel = _make_apply_ufunc_kernel(method, x_values)
        out_w = xr.apply_ufunc(
            kernel,
            da_w,
            input_core_dims=[[sdim]],
            output_core_dims=[[sdim]],
            dask="parallelized",
            output_dtypes=[da_w.dtype],
        )
    else:
        spectra_2d = da_w.values.reshape(-1, da_w.shape[-1])
        out_2d = _normalize_numpy_block(spectra_2d, method, x_values)
        packed = reshape_row_stack_to(out_2d, da_w.shape)
        out_w = da_w.copy(data=packed)

    if tuple(out_w.dims) != orig_order:
        out_w = out_w.transpose(*orig_order)

    return with_new_values(da, out_w.data, "normalization", {"method": method})

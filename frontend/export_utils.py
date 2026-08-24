# -*- coding: utf-8 -*-
"""Helpers that pack analysis results into .npz bytes for st.download_button."""

from __future__ import annotations

import io

import numpy as np
import xarray as xr


def spectra_to_npz(da: xr.DataArray, excluded_mask: np.ndarray | None = None) -> bytes:
    """Pack the processed cube at full resolution and original shape.

    ``excluded_mask`` (bool over the spatial dims, True = manually excluded) is
    written alongside when supplied, so a consumer can tell hand-excluded
    spectra from the ones Clean Data dropped — both are all-NaN rows in
    ``data``, and ``data`` always keeps the raw WDF geometry either way.
    """
    arrays = {"data": da.values}
    for dim in da.dims:
        if dim in da.coords:
            arrays[f"{dim}_coords"] = da.coords[dim].values
    if excluded_mask is not None and np.any(excluded_mask):
        arrays["excluded_mask"] = np.asarray(excluded_mask, dtype=bool)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def mean_spectrum_to_npz(da: xr.DataArray) -> bytes:
    """Pack nan-mean over all non-spectral dims into a 1-D .npz."""
    spectral_dim = da.dims[-1]
    non_spectral = [d for d in da.dims if d != spectral_dim]
    if non_spectral:
        mean = np.nanmean(da.values, axis=tuple(range(len(non_spectral))))
    else:
        mean = da.values.copy()
    arrays = {"mean": mean}
    if spectral_dim in da.coords:
        arrays[f"{spectral_dim}_coords"] = da.coords[spectral_dim].values
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def nmf_to_npz(nmf_result: dict) -> bytes:
    ab = nmf_result["abundances"]
    arrays = {
        "components":    nmf_result["components"],
        "spectral_axis": nmf_result["spectral_coords"],
        "abundances":    ab.values,
    }
    for dim in ab.dims[:-1]:  # spatial dims, skip "component"
        if dim in ab.coords:
            arrays[f"{dim}_coords"] = ab.coords[dim].values
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def mcr_to_npz(mcr_result: dict) -> bytes:
    """Pack an MCR-ALS result: pure spectra (S), concentration maps (C),
    lack-of-fit, and the per-component rotational-ambiguity bands."""
    ab = mcr_result["abundances"]
    meta = mcr_result.get("meta", {})
    arrays = {
        "components":    mcr_result["components"],   # S (k, n_spectral)
        "spectral_axis": mcr_result["spectral_coords"],
        "concentrations": ab.values,                 # C (spatial..., k)
        "lof":           np.asarray(meta.get("lof", np.nan)),
    }
    for dim in ab.dims[:-1]:  # spatial dims, skip "component"
        if dim in ab.coords:
            arrays[f"{dim}_coords"] = ab.coords[dim].values
    amb = mcr_result.get("ambiguity")
    if amb and amb.get("ok"):
        arrays["ambiguity_f_min"] = np.asarray(amb["f_min"])
        arrays["ambiguity_f_max"] = np.asarray(amb["f_max"])
        arrays["ambiguity_f_range"] = np.asarray(amb["f_range"])
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def fit_curves_to_npz(fit_result) -> bytes:
    arrays = {
        "spectral_axis": fit_result.x,
        "y_data":        fit_result.y_data,
        "y_fit":         fit_result.y_fit,
        "residual":      fit_result.residual,
    }
    for band in fit_result.bands:
        arrays[f"band_{band.label.replace(' ', '_')}"] = band.curve
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def batch_fit_to_npz(batch_result, labels: list[str], row_coords, col_coords) -> bytes:
    arrays = {
        "row_coords":         row_coords,
        "col_coords":         col_coords,
        "r_squared":          batch_result.r_squared_map,
        "reduced_chi_square": batch_result.reduced_chi_square_map,
        "success":            batch_result.success_map,
    }
    for label in labels:
        for param, arr in batch_result.band_results[label].items():
            arrays[f"{label.replace(' ', '_')}_{param}"] = arr
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()

# -*- coding: utf-8 -*-
"""Helpers that pack analysis results into .npz bytes for st.download_button."""

from __future__ import annotations

import io

import numpy as np
import xarray as xr


def spectra_to_npz(da: xr.DataArray) -> bytes:
    arrays = {"data": da.values}
    for dim in da.dims:
        if dim in da.coords:
            arrays[f"{dim}_coords"] = da.coords[dim].values
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

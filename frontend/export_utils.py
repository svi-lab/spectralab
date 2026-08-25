"""Helpers that pack analysis results into .npz / .csv bytes for st.download_button.

Two formats per result, aimed at two audiences:

* ``.csv`` — plain-text tables that open directly in Origin / Excel / any
  plotting tool, for publication figures. Spectra export in *wide* layout
  (first column = spectral axis, one column per spectrum, 1-based ``r{i}_c{j}``
  headers matching the app's display numbering); maps export in *long* layout
  (one row per pixel).
* ``.npz`` — NumPy archives bundling everything (full precision, coordinates,
  masks) for programmatic use via ``numpy.load``.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import xarray as xr

# Compact but full-precision-enough text for plotting; keeps the biggest map
# exports a few hundred MB instead of a GB of digits.
_CSV_FLOAT_FORMAT = "%.8g"


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


def _spectrum_headers(da: xr.DataArray) -> list[str]:
    """One header per spectrum, in C-order, using the app's 1-based display
    numbering (``r{row}_c{col}`` for maps, ``s{n}`` for line scans)."""
    spectral_dim = da.dims[-1]
    spatial = [d for d in da.dims if d != spectral_dim]
    if len(spatial) == 2:
        n_col = da.sizes[spatial[1]]
        return [f"r{r + 1}_c{c + 1}" for r in range(da.sizes[spatial[0]]) for c in range(n_col)]
    if len(spatial) == 1:
        return [f"s{i + 1}" for i in range(da.sizes[spatial[0]])]
    return ["intensity"]


def _spectral_axis(da: xr.DataArray) -> np.ndarray:
    spectral_dim = da.dims[-1]
    if spectral_dim in da.coords:
        return da.coords[spectral_dim].values
    return np.arange(da.shape[-1])


def spectra_to_csv(da: xr.DataArray, axis_label: str) -> bytes:
    """Wide CSV of every spectrum: first column = spectral axis, one column
    per spectrum. Removed/excluded spectra stay as empty cells, so column
    count always matches the raw scan geometry."""
    values = da.values.reshape(-1, da.shape[-1])
    df = pd.DataFrame(values.T, columns=_spectrum_headers(da))
    df.insert(0, axis_label, _spectral_axis(da))
    return df.to_csv(index=False, float_format=_CSV_FLOAT_FORMAT).encode("utf-8")


def mean_spectrum_to_csv(da: xr.DataArray, axis_label: str) -> bytes:
    """Two-column CSV: spectral axis + nan-mean over all spectra."""
    spectral_dim = da.dims[-1]
    non_spectral = [d for d in da.dims if d != spectral_dim]
    if non_spectral:
        mean = np.nanmean(da.values, axis=tuple(range(len(non_spectral))))
    else:
        mean = da.values
    df = pd.DataFrame({axis_label: _spectral_axis(da), "mean_intensity": mean})
    return df.to_csv(index=False, float_format=_CSV_FLOAT_FORMAT).encode("utf-8")


def components_to_csv(
    components: np.ndarray,
    spectral_coords: np.ndarray,
    axis_label: str,
) -> bytes:
    """Component spectra (NMF / MCR) as columns against the spectral axis."""
    df = pd.DataFrame(
        components.T,
        columns=[f"component_{i + 1}" for i in range(components.shape[0])],
    )
    df.insert(0, axis_label, np.asarray(spectral_coords))
    return df.to_csv(index=False, float_format=_CSV_FLOAT_FORMAT).encode("utf-8")


def abundance_maps_to_csv(abundances: xr.DataArray, value_name: str = "abundance") -> bytes:
    """Long CSV of the component maps: one row per pixel, one value column per
    component. ``row``/``column`` are 1-based pixel indices (the app's display
    numbering); ``y_um``/``x_um`` are the stage coordinates when present."""
    spatial = [d for d in abundances.dims if d != "component"]
    k = abundances.sizes["component"]
    flat = abundances.transpose(*spatial, "component").values.reshape(-1, k)

    data: dict[str, np.ndarray] = {}
    if len(spatial) == 2:
        n_row, n_col = abundances.sizes[spatial[0]], abundances.sizes[spatial[1]]
        r_idx, c_idx = np.divmod(np.arange(n_row * n_col), n_col)
        data["row"] = r_idx + 1
        data["column"] = c_idx + 1
        if spatial[0] in abundances.coords:
            data["y_um"] = abundances.coords[spatial[0]].values[r_idx]
        if spatial[1] in abundances.coords:
            data["x_um"] = abundances.coords[spatial[1]].values[c_idx]
    else:
        data["spectrum"] = np.arange(flat.shape[0]) + 1
    for i in range(k):
        data[f"{value_name}_{i + 1}"] = flat[:, i]
    df = pd.DataFrame(data)
    return df.to_csv(index=False, float_format=_CSV_FLOAT_FORMAT).encode("utf-8")


def scalar_map_to_csv(
    z: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    value_name: str,
) -> bytes:
    """Long CSV of one 2-D scalar map: one row per pixel (1-based ``row`` /
    ``column`` indices + µm coordinates + the value)."""
    n_row, n_col = z.shape
    r_idx, c_idx = np.divmod(np.arange(n_row * n_col), n_col)
    df = pd.DataFrame(
        {
            "row": r_idx + 1,
            "column": c_idx + 1,
            "y_um": np.asarray(row_coords)[r_idx],
            "x_um": np.asarray(col_coords)[c_idx],
            value_name: z.reshape(-1),
        }
    )
    return df.to_csv(index=False, float_format=_CSV_FLOAT_FORMAT).encode("utf-8")


def nmf_to_npz(nmf_result: dict) -> bytes:
    ab = nmf_result["abundances"]
    arrays = {
        "components": nmf_result["components"],
        "spectral_axis": nmf_result["spectral_coords"],
        "abundances": ab.values,
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
        "components": mcr_result["components"],  # S (k, n_spectral)
        "spectral_axis": mcr_result["spectral_coords"],
        "concentrations": ab.values,  # C (spatial..., k)
        "lof": np.asarray(meta.get("lof", np.nan)),
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
        "y_data": fit_result.y_data,
        "y_fit": fit_result.y_fit,
        "residual": fit_result.residual,
    }
    for band in fit_result.bands:
        arrays[f"band_{band.label.replace(' ', '_')}"] = band.curve
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def batch_fit_to_npz(batch_result, labels: list[str], row_coords, col_coords) -> bytes:
    arrays = {
        "row_coords": row_coords,
        "col_coords": col_coords,
        "r_squared": batch_result.r_squared_map,
        "reduced_chi_square": batch_result.reduced_chi_square_map,
        "success": batch_result.success_map,
    }
    for label in labels:
        for param, arr in batch_result.band_results[label].items():
            arrays[f"{label.replace(' ', '_')}_{param}"] = arr
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()

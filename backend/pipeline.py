"""Processing pipeline: file loading and staged wdfkit transformations."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from wdfkit import WDFReader

from _shared._spectral import (
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from _shared.clean_data import CleanData
from _shared.dataset import SpectralDataset, validate_spectral_dataset
from _shared.normalize import normalize
from cosmic_ray import CosmicRayRemover
from spectra_cleaner import Denoiser
from spectra_smoother import SpectraSmoother

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_wdf(file_bytes: bytes) -> SpectralDataset:
    """Parse a .wdf file from raw bytes and return a SpectralDataset.

    WDFReader requires a real file path. The bytes are written to a temp file
    in the app's own directory (avoids corporate AV/DLP quarantine of %TEMP%).
    """
    tmp_path = Path(__file__).parent / f"_wdf_{uuid.uuid4().hex}.wdf"
    try:
        tmp_path.write_bytes(file_bytes)
        da, pil_image = WDFReader(str(tmp_path))
        da.load()
    finally:
        tmp_path.unlink(missing_ok=True)

    # ── Extract image + EXIF ────────────────────────────────────────────
    image_arr: np.ndarray | None = None
    image_meta: dict[str, Any] | None = None

    if pil_image is not None:
        image_arr = np.array(pil_image.convert("RGB"))
        w_px, h_px = pil_image.size
        try:
            raw_exif = pil_image.getexif()
            # 65184 (0xFEA0) is a Renishaw top-level tag: (origin_x_µm, origin_y_µm)
            origin_xy = raw_exif.get(65184)
            # 41486/41487 (FocalPlaneXResolution/YResolution) are in the ExifIFD sub-IFD
            exif_ifd = raw_exif.get_ifd(34665)
            fov_x_um = exif_ifd.get(41486)
            fov_y_um = exif_ifd.get(41487)
            if origin_xy and fov_x_um and fov_y_um:
                image_meta = {
                    "origin_x": float(origin_xy[0]),
                    "origin_y": float(origin_xy[1]),
                    "fov_x": float(fov_x_um),
                    "fov_y": float(fov_y_um),
                    "width_px": w_px,
                    "height_px": h_px,
                }
        except Exception as exc:
            warnings.warn(f"EXIF extraction failed: {exc}", stacklevel=2)

    # ── Laser wavelength ────────────────────────────────────────────────
    laser_nm: float | None = None
    for _attr in (
        "laser_wavelength_nm",
        "LaserWavelength",
        "laser_wavelength",
        "ExcitationWavelength",
        "LaserWaveLength",
        "excitation_wavelength",
    ):
        _val = da.attrs.get(_attr)
        if _val is not None:
            try:
                laser_nm = float(_val)
                break
            except (TypeError, ValueError):
                pass

    # ── Spectral axis ───────────────────────────────────────────────────
    spectral_dim = da.dims[-1]
    spectral_vals = da.coords[spectral_dim].values
    spec_min = float(spectral_vals.min())
    spec_max = float(spectral_vals.max())
    spectral_unit: str = da.coords[spectral_dim].attrs.get("units", "")
    spectral_units: str = da.attrs.get("spectral_units", "")

    # ── Scalar metadata ─────────────────────────────────────────────────
    def _float_attr(key: str) -> float:
        v = da.attrs.get(key)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    is_valid, validation_msg = validate_spectral_dataset(da, spectral_units)

    return SpectralDataset(
        da=da,
        spectral_dim=spectral_dim,
        spectral_units=spectral_units,
        spectral_unit=spectral_unit,
        spec_min=spec_min,
        spec_max=spec_max,
        laser_nm=laser_nm,
        is_map=da.ndim == 3 and "row" in da.dims and "column" in da.dims,
        image_arr=image_arr,
        image_meta=image_meta,
        laser_power=_float_attr("laser_power"),
        exposure_time=_float_attr("exposure_time"),
        comment=da.attrs.get("comment") or "",
        dims=da.dims,
        shape=da.shape,
        ndim=da.ndim,
        is_valid=is_valid,
        validation_msg=validation_msg,
    )


# ---------------------------------------------------------------------------
# Pipeline stages — pure per-stage functions
#
# Each stage takes a DataArray plus its own primitive param dict and returns a
# NEW DataArray. The frontend wraps each of these in its own st.cache_data
# entry (frontend/pipeline_cache.py) so a downstream param change never
# recomputes upstream stages; preprocess() below chains them directly for
# non-cached/test use.
#
# Dtype contract: the pipeline never intentionally changes precision — only
# the WDFReader ``dtype`` parameter does (float64 by default; float32 halves
# RAM end-to-end when the user flips it). Several numpy ops upcast silently
# along the way (e.g. a float64 spectral-coordinate array in the area-norm
# trapezoid, or a bare Python float scale constant) — ``_restore_dtype``
# casts back so those internal upcasts never leak into the stored result.
# ---------------------------------------------------------------------------


def _restore_dtype(out: xr.DataArray, in_dtype: np.dtype) -> xr.DataArray:
    """Cast ``out`` back to ``in_dtype`` if a stage silently upcast it."""
    if out.dtype != in_dtype:
        return out.astype(in_dtype)
    return out


def stage_normalize(da: xr.DataArray, norm_params: dict) -> xr.DataArray:
    out = normalize(da, method=norm_params["method"])
    return _restore_dtype(out, da.dtype)


def stage_clean_data(da: xr.DataArray, cd_params: dict) -> xr.DataArray:
    cleaned = CleanData(n_zeros=cd_params["n_zeros"]).check(da)
    # Re-pad rows dropped from 2-D line scans as NaN so every stage keeps the
    # raw shape (3-D maps already NaN-fill in place) — downstream PCA/NMF/fit
    # code handles NaN rows generically.
    for dim in cleaned.dims:
        if dim in da.coords and cleaned.sizes[dim] != da.sizes[dim]:
            cleaned = cleaned.reindex({dim: da.coords[dim]})
    return cleaned


def stage_cosmic_ray_removal(da: xr.DataArray, crr_params: dict) -> xr.DataArray:
    crr = CosmicRayRemover(
        spike_width=crr_params["spike_width"],
        spike_threshold=crr_params["spike_threshold"],
        spike_passes=crr_params["spike_passes"],
        broad_spike_width=crr_params.get("broad_spike_width", 15),
        force_1d=crr_params.get("force_1d", False),
        map_sensitivity=crr_params["map_sensitivity"],
        map_disk_radius=crr_params["map_disk_radius"],
        map_spike_width=crr_params["map_spike_width"],
        map_method=crr_params["map_method"],
        map_n_components=crr_params["map_n_components"],
    )
    out = crr.remove(da)
    return _restore_dtype(out, da.dtype)


def stage_denoise(da: xr.DataArray, denoise_params: dict) -> xr.DataArray:
    n_components = _parse_n_components(denoise_params)

    smoother = None
    if denoise_params["per_spectrum"]:
        sm_p = denoise_params["smoother"]
        smoother = SpectraSmoother(
            method=sm_p["method"],
            window_length=sm_p["window_length"],
            polyorder=sm_p["polyorder"],
            lam=sm_p.get("lam"),
            d=sm_p["d"],
            auto_lam_calls=sm_p["auto_lam_calls"],
            wavelet=sm_p.get("wavelet", "db4"),
            wavelet_level=sm_p.get("wavelet_level"),
            wavelet_threshold=sm_p.get("wavelet_threshold", "soft"),
        )

    denoiser = Denoiser(
        n_components=n_components,
        subtract_min=denoise_params["subtract_min"],
        restore_min=denoise_params["restore_min"],
        per_spectrum=denoise_params["per_spectrum"],
        smoother=smoother,
    )
    # PCA still fits float64 internally (sklearn) — only the reconstruction
    # written back into the DataArray needs casting back.
    out = denoiser.denoise(da)
    return _restore_dtype(out, da.dtype)


def stage_exclude(
    da: xr.DataArray,
    mask: np.ndarray,
    spectral_dim: str | None = None,
) -> xr.DataArray:
    """NaN out manually excluded spectra, preserving the array structure.

    ``mask`` is a boolean array over the *spatial* dims only (True = excluded):
    ``(n_row, n_col)`` for a map, ``(n_point,)`` for a line scan. Flat index is
    C-order ``i = r * n_col + c`` — the same convention used by
    ``_shared._factorization._flatten_to_row_stack``, ``CleanData._handle_3d``
    and ``_shared.scan_geometry``'s meshgrid ravel.

    Excluded spectra become all-NaN rows *in place*: shape, dims and coords are
    identical to the input, so the original WDF geometry survives into exports
    and every index-based downstream analysis. Downstream code already treats
    all-NaN rows as invalid (PCA, NMF/MCR, fit_map_gaussian, the charts).

    This runs as the final pipeline stage — see ``apply_exclusion`` /
    ``preprocess`` below and ``frontend/pipeline_cache.py``, which applies it
    on top of the memoized pre-exclusion result so editing the mask never
    invalidates the CRR / denoise caches.
    """
    sdim = resolve_spectral_dim(da, spectral_dim)
    da_t, orig_order = transpose_spectral_last(da, sdim)
    spatial_dims = tuple(d for d in da_t.dims if d != sdim)

    mask = np.asarray(mask, dtype=bool)
    expected = tuple(da_t.sizes[d] for d in spatial_dims)
    if mask.shape != expected:
        raise ValueError(
            f"exclusion mask shape {mask.shape} does not match the spatial "
            f"shape {expected} of dims {spatial_dims}"
        )
    if not mask.any():
        return da

    # np.nan is a Python float — np.where would upcast a float32 array, hence
    # the _restore_dtype below (the no-silent-upcast contract above).
    values = np.where(mask[..., None], np.nan, np.asarray(da_t.values))

    flat = np.flatnonzero(mask.ravel())
    out = with_new_values(
        da_t,
        values,
        "Excluded Spectra",
        {
            "n_excluded": int(flat.size),
            "n_total": int(mask.size),
            "spatial_dims": list(spatial_dims),
            "flat_indices": flat.tolist(),
        },
    )
    if tuple(out.dims) != orig_order:
        out = out.transpose(*orig_order)
    return _restore_dtype(out, da.dtype)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

# (var_name, human-readable label, DataArray) — in run order
StageRecord = tuple[str, str, xr.DataArray]


def assemble_dataset(
    dataset: SpectralDataset,
    stage_records: list[StageRecord],
    keep_stages: bool = True,
) -> xr.Dataset:
    """Bundle stage results into one xr.Dataset — the per-file result object.

    Attrs always record the full provenance (``stage_vars`` in run order,
    ``stage_labels`` var→label, ``final_var``); with ``keep_stages=False``
    only the final data variable is stored, except ``clean_data`` is also
    kept when that stage ran — the Clean Data tab's removal grid reads it.
    """
    if keep_stages:
        records = stage_records
    else:
        records = [stage_records[-1]]
        cd_rec = next((r for r in stage_records if r[0] == "clean_data"), None)
        if cd_rec is not None and cd_rec[0] != records[-1][0]:
            records = [cd_rec, records[-1]]
    ds = xr.Dataset({var: da for var, _label, da in records})
    ds.attrs["stage_vars"] = [var for var, _, _ in stage_records]
    ds.attrs["stage_labels"] = {var: label for var, label, _ in stage_records}
    ds.attrs["final_var"] = stage_records[-1][0]
    return ds


# ---------------------------------------------------------------------------
# Stage chain — single sequencer (cached frontend and preprocess() both use this)
# ---------------------------------------------------------------------------

RecipeFn = Callable[[xr.DataArray, dict[str, Any]], xr.DataArray]


def run_stage_chain(
    dataset: SpectralDataset,
    params: dict[str, Any],
    keep_stages: bool = True,
    *,
    cosmic_ray: RecipeFn | None = None,
    denoise: RecipeFn | None = None,
) -> xr.Dataset:
    """Normalize → CleanData → CRR → Denoise. Does not apply exclusion.

    ``cosmic_ray`` / ``denoise`` receive ``(da, recipe)`` where ``recipe`` is
    the cumulative param dict up to (and including) that stage — the frontend
    cache keys off this snapshot. Defaults call the matching ``stage_*`` with
    the stage's own subdict.
    """
    apply_crr = cosmic_ray or (lambda da, rec: stage_cosmic_ray_removal(da, rec["crr"]))
    apply_denoise = denoise or (lambda da, rec: stage_denoise(da, rec["denoise"]))

    da: xr.DataArray = dataset.da
    stage_records: list[StageRecord] = []
    recipe: dict[str, Any] = {}

    recipe["cd_enabled"] = bool(params.get("cd_enabled"))
    recipe["cd"] = params.get("cd", {})
    recipe["norm1_enabled"] = bool(params.get("norm1_enabled"))
    recipe["norm1"] = params.get("norm1", {})

    # ── CleanData (raw) ────────────────────────────────────────────────
    # Runs on raw ADC values *before* normalization. min_max normalization
    # sets every spectrum's minimum to exactly 0, so detecting consecutive
    # zeros after norm would flag every spectrum at n_zeros=1.
    if not recipe["norm1_enabled"] and recipe["cd_enabled"]:
        stage_records.append(("raw", "Raw", dataset.da))
    if recipe["cd_enabled"]:
        da = stage_clean_data(da, recipe["cd"])
        stage_records.append(("clean_data", "Clean Data", da))

    # ── Normalization 1 ────────────────────────────────────────────────
    if recipe["norm1_enabled"]:
        da = stage_normalize(da, recipe["norm1"])
        stage_records.append(("norm_before", "Normalized (before)", da))
    elif not stage_records:
        stage_records.append(("raw", "Raw", dataset.da))

    # ── CosmicRayRemover ───────────────────────────────────────────────
    # Recipe grows even when the stage is off so a later cached stage's key
    # still includes every upstream flag (same as the old frontend copy).
    recipe["crr_enabled"] = bool(params.get("crr_enabled"))
    recipe["crr"] = params.get("crr", {})
    if recipe["crr_enabled"]:
        da = apply_crr(da, dict(recipe))
    recipe["norm2_enabled"] = bool(params.get("norm2_enabled"))
    recipe["norm2"] = params.get("norm2", {})
    if recipe["crr_enabled"]:
        if recipe["norm2_enabled"]:
            da = stage_normalize(da, recipe["norm2"])
            stage_records.append(("norm_post_crr", "Normalized (after cosmic rays)", da))
        else:
            stage_records.append(("crr", "Cosmic rays removed", da))

    # ── Denoiser ───────────────────────────────────────────────────────
    recipe["denoise_enabled"] = bool(params.get("denoise_enabled"))
    recipe["denoise"] = params.get("denoise", {})
    if recipe["denoise_enabled"]:
        da = apply_denoise(da, dict(recipe))
    recipe["norm3_enabled"] = bool(params.get("norm3_enabled"))
    recipe["norm3"] = params.get("norm3", {})
    if recipe["denoise_enabled"]:
        if recipe["norm3_enabled"]:
            da = stage_normalize(da, recipe["norm3"])
            stage_records.append(("norm_post_denoise", "Normalized (after denoising)", da))
        else:
            stage_records.append(("denoised", "Denoised", da))

    return assemble_dataset(dataset, stage_records, keep_stages=keep_stages)


def apply_exclusion(
    ds_pre: xr.Dataset,
    mask: np.ndarray,
    spectral_dim: str,
    keep_stages: bool,
) -> xr.Dataset:
    """Append the manual-exclusion stage on top of a pre-exclusion Dataset.

    ``ds_pre`` is a shared memo reference in the app — its attrs dicts are
    copied, never extended in place.
    """
    masked = stage_exclude(ds_pre[ds_pre.attrs["final_var"]], mask, spectral_dim)

    if keep_stages:
        ds = ds_pre.assign({"excluded": masked})
    else:
        extras: dict[str, xr.DataArray] = {}
        if "clean_data" in ds_pre.data_vars:
            extras["clean_data"] = ds_pre["clean_data"]
        ds = xr.Dataset({"excluded": masked, **extras})
    ds.attrs["stage_vars"] = [*ds_pre.attrs["stage_vars"], "excluded"]
    ds.attrs["stage_labels"] = {**ds_pre.attrs["stage_labels"], "excluded": "Excluded"}
    ds.attrs["final_var"] = "excluded"
    return ds


def preprocess(
    dataset: SpectralDataset,
    params: dict[str, Any],
    keep_stages: bool = True,
) -> xr.Dataset:
    """Run the full preprocessing pipeline and return one xr.Dataset.

    Same sequencer as the Streamlit app (``run_stage_chain``); the app injects
    cached CRR/denoise callables. Manual exclusion is read from
    ``params["excl"]["mask"]`` — a single boolean spatial mask. The frontend's
    equivalent key is ``params["excl"]["masks"]``, a ``{filename: mask}`` dict.
    """
    ds = run_stage_chain(dataset, params, keep_stages)
    excl_mask = (params.get("excl") or {}).get("mask")
    if excl_mask is not None and np.any(excl_mask):
        ds = apply_exclusion(ds, excl_mask, dataset.spectral_dim, keep_stages)
    return ds


def _parse_n_components(sc_p: dict) -> int | float | str | None:
    nc_type = sc_p["n_components_type"]
    if nc_type == "mle":
        return "mle"
    if nc_type == "None":
        return None
    if nc_type == "int":
        return int(sc_p["n_components_int"])
    if nc_type == "float":
        return float(sc_p["n_components_float"])
    return "mle"

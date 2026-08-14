"""Processing pipeline: file loading and staged wdfkit transformations."""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from wdfkit import WDFReader

from cosmic_ray import CosmicRayRemover
from spectra_cleaner import Denoiser
from spectra_smoother import SpectraSmoother
from _shared.clean_data import CleanData
from _shared.normalize import normalize
from _shared.dataset import SpectralDataset, validate_spectral_dataset
from _shared._spectral import (
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
# Background suppression is currently disabled app-wide — see
# stage_background_suppress() below. Kept importable for a future re-enable.
# from background import BackgroundSuppressor
# from background._scale import interp_reference


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
            exif_ifd  = raw_exif.get_ifd(34665)
            fov_x_um  = exif_ifd.get(41486)
            fov_y_um  = exif_ifd.get(41487)
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
    for _attr in ("laser_wavelength_nm", "LaserWavelength", "laser_wavelength",
                  "ExcitationWavelength", "LaserWaveLength", "excitation_wavelength"):
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
        da             = da,
        spectral_dim   = spectral_dim,
        spectral_units = spectral_units,
        spectral_unit  = spectral_unit,
        spec_min       = spec_min,
        spec_max       = spec_max,
        laser_nm       = laser_nm,
        is_map         = da.ndim == 3 and "row" in da.dims and "column" in da.dims,
        image_arr      = image_arr,
        image_meta     = image_meta,
        laser_power    = _float_attr("laser_power"),
        exposure_time  = _float_attr("exposure_time"),
        comment        = da.attrs.get("comment") or "",
        dims           = da.dims,
        shape          = da.shape,
        ndim           = da.ndim,
        is_valid       = is_valid,
        validation_msg = validation_msg,
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

    This runs as the final pipeline stage — see ``preprocess`` below and
    ``frontend/pipeline_cache.py``, which applies it on top of the memoized
    pre-exclusion result so editing the mask never invalidates the CRR /
    denoise caches.
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


# Background suppression is currently disabled app-wide (no UI entry point
# calls this) — kept for a future re-enable. Fixed here: the reference used
# to be cast to da.dtype *before* the interpolation branch, so a mismatched-
# length reference got resampled at reduced precision when the reader runs
# in float32 mode; interpolation now happens at full precision and only the
# result is cast down.
# def stage_background_suppress(
#     da: xr.DataArray, bg_params: dict, spectral_dim: str
# ) -> xr.DataArray:
#     reference = np.asarray(bg_params["reference"], dtype=float)
#     spectral_coords = da.coords[spectral_dim].values
#
#     # Interpolate reference onto the data's spectral axis if needed
#     if len(reference) != len(spectral_coords):
#         ref_x = np.asarray(bg_params["reference_x"], dtype=float)
#         # np.interp always returns float64 — cast back to the data's dtype.
#         reference, _ = interp_reference(ref_x, reference, spectral_coords)
#     reference = reference.astype(da.dtype, copy=False)
#
#     suppressor = BackgroundSuppressor(
#         reference=reference,
#         spectral_dim=spectral_dim,
#         fixed_scale=bg_params["fixed_scale"],
#     )
#     da_out, _bg_meta = suppressor.suppress(da)
#     return _restore_dtype(da_out, da.dtype)


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
    only the final data variable is stored (memory: analysis pages never
    look at intermediates).
    """
    records = stage_records if keep_stages else stage_records[-1:]
    ds = xr.Dataset({var: da for var, _label, da in records})
    ds.attrs["stage_vars"] = [var for var, _, _ in stage_records]
    ds.attrs["stage_labels"] = {var: label for var, label, _ in stage_records}
    ds.attrs["final_var"] = stage_records[-1][0]
    return ds


# ---------------------------------------------------------------------------
# Preprocessing pipeline (single-shot, non-cached)
# ---------------------------------------------------------------------------

def preprocess(
    dataset: SpectralDataset,
    params: dict[str, Any],
    keep_stages: bool = True,
) -> xr.Dataset:
    """Run the full preprocessing pipeline and return one xr.Dataset.

    Non-cached convenience wrapper chaining the stage_* functions — the
    Streamlit app routes through frontend/pipeline_cache.py instead, which
    runs the same sequencing with a per-stage cache between steps.

    Manual exclusion is read from ``params["excl"]["mask"]`` — a single
    boolean spatial mask, since this entry point processes one dataset. The
    frontend's equivalent key is ``params["excl"]["masks"]``, a
    ``{filename: mask}`` dict, because get_finals loops over every loaded file.
    """
    da: xr.DataArray = dataset.da
    stage_records: list[StageRecord] = []

    # Background suppression is currently disabled app-wide (see
    # stage_background_suppress above), so normalization is never deferred.
    # When bg_enabled: subtraction runs on data after all selected
    # preprocessing steps but WITHOUT min-max/area normalization (which is
    # data-dependent and would invalidate the physics scale and distort the
    # fitted one) — the user's chosen normalization still runs, deferred
    # until after the subtraction.
    # defer_norm = bool(params.get("bg_enabled"))
    defer_norm = False

    # ── Normalization 1 (raw) ──────────────────────────────────────────
    if params.get("norm1_enabled") and not defer_norm:
        da = stage_normalize(da, params["norm1"])
        stage_records.append(("norm_before", "Normalized (raw)", da))
    else:
        stage_records.append(("raw", "Raw", dataset.da))

    # ── CleanData ──────────────────────────────────────────────────────
    if params.get("cd_enabled"):
        da = stage_clean_data(da, params["cd"])
        stage_records.append(("clean_data", "Clean Data", da))

    # ── CosmicRayRemover ───────────────────────────────────────────────
    if params.get("crr_enabled"):
        da = stage_cosmic_ray_removal(da, params["crr"])
        if params.get("norm2_enabled") and not defer_norm:
            da = stage_normalize(da, params["norm2"])
            stage_records.append(("norm_post_crr", "Normalized (post-CR)", da))
        else:
            stage_records.append(("crr", "CR Removed", da))

    # ── Denoiser ───────────────────────────────────────────────────────
    if params.get("denoise_enabled"):
        da = stage_denoise(da, params["denoise"])
        if params.get("norm3_enabled") and not defer_norm:
            da = stage_normalize(da, params["norm3"])
            stage_records.append(("norm_post_denoise", "Normalized (final)", da))
        else:
            stage_records.append(("denoised", "Denoised", da))

    # ── Background Suppression (disabled app-wide — see stage_background_suppress) ──
    # if params.get("bg_enabled"):
    #     da = stage_background_suppress(da, params["bg"], dataset.spectral_dim)
    #     stage_records.append(("bg_removed", "Background removed", da))
    #
    #     # Deferred normalization: apply the user's chosen method now that the
    #     # subtraction has happened in raw intensity space.
    #     if defer_norm and (params.get("norm1_enabled") or params.get("norm2_enabled")
    #                        or params.get("norm3_enabled")):
    #         norm_p = params.get("norm1") or params.get("norm2") or params.get("norm3") or {}
    #         if norm_p.get("method"):
    #             da = stage_normalize(da, norm_p)
    #             stage_records.append(("norm_post_bg", "Normalized (post-suppression)", da))

    # ── Manual exclusion (always last) ─────────────────────────────────
    excl_mask = (params.get("excl") or {}).get("mask")
    if excl_mask is not None and np.any(excl_mask):
        da = stage_exclude(da, excl_mask, dataset.spectral_dim)
        stage_records.append(("excluded", "Excluded", da))

    return assemble_dataset(dataset, stage_records, keep_stages=keep_stages)


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

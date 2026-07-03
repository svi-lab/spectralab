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
from background import BackgroundSuppressor
from background._scale import interp_reference


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
# Normalization helper
# ---------------------------------------------------------------------------

def _apply_normalize(da: xr.DataArray, norm_params: dict) -> xr.DataArray:
    return normalize(da, method=norm_params["method"])


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    dataset: SpectralDataset,
    params: dict[str, Any],
) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """Run the full preprocessing pipeline and return (stages, da_final).

    ``stages`` is an ordered dict mapping a human-readable label to the
    DataArray at each point — used directly by the Progress tab.
    """
    da: xr.DataArray = dataset.da
    stages: dict[str, xr.DataArray] = {}

    # Background suppression always operates outside normalization: the
    # subtraction runs on data after all selected preprocessing steps but
    # WITHOUT min-max/area normalization (which is data-dependent and would
    # invalidate the physics scale and distort the fitted one). The user's
    # chosen normalization still runs — deferred until after the subtraction.
    defer_norm = bool(params.get("bg_enabled"))

    # ── Normalization 1 (raw) ──────────────────────────────────────────
    if params.get("norm1_enabled") and not defer_norm:
        da = _apply_normalize(da, params["norm1"])
        stages["Normalized (raw)"] = da
    else:
        stages["Raw"] = dataset.da

    # ── CleanData ──────────────────────────────────────────────────────
    if params.get("cd_enabled"):
        cd_p = params["cd"]
        da = CleanData(n_zeros=cd_p["n_zeros"]).check(da)
        stages["Clean Data"] = da

    # ── CosmicRayRemover ───────────────────────────────────────────────
    if params.get("crr_enabled"):
        crr_p = params["crr"]
        crr = CosmicRayRemover(
            spike_width=crr_p["spike_width"],
            spike_threshold=crr_p["spike_threshold"],
            spike_passes=crr_p["spike_passes"],
            broad_spike_width=crr_p.get("broad_spike_width", 15),
            force_1d=crr_p.get("force_1d", False),
            map_sensitivity=crr_p["map_sensitivity"],
            map_disk_radius=crr_p["map_disk_radius"],
            map_spike_width=crr_p["map_spike_width"],
            map_method=crr_p["map_method"],
            map_n_components=crr_p["map_n_components"],
        )
        da = crr.remove(da)
        if params.get("norm2_enabled") and not defer_norm:
            da = _apply_normalize(da, params["norm2"])
            stages["Normalized (post-CR)"] = da
        else:
            stages["CR Removed"] = da

    # ── Denoiser ───────────────────────────────────────────────────────
    if params.get("denoise_enabled"):
        sc_p = params["denoise"]
        n_components = _parse_n_components(sc_p)

        smoother = None
        if sc_p["per_spectrum"]:
            sm_p = sc_p["smoother"]
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
            subtract_min=sc_p["subtract_min"],
            restore_min=sc_p["restore_min"],
            per_spectrum=sc_p["per_spectrum"],
            smoother=smoother,
        )
        da = denoiser.denoise(da)
        if params.get("norm3_enabled") and not defer_norm:
            da = _apply_normalize(da, params["norm3"])
            stages["Normalized (final)"] = da
        else:
            stages["Denoised"] = da

    # ── Background Suppression ─────────────────────────────────────────
    if params.get("bg_enabled"):
        bg_p = params["bg"]
        reference = np.asarray(bg_p["reference"], dtype=float)
        spectral_coords = da.coords[dataset.spectral_dim].values

        # Interpolate reference onto the data's spectral axis if needed
        if len(reference) != len(spectral_coords):
            ref_x = np.asarray(bg_p["reference_x"], dtype=float)
            reference, _ = interp_reference(ref_x, reference, spectral_coords)

        suppressor = BackgroundSuppressor(
            reference=reference,
            spectral_dim=dataset.spectral_dim,
            fixed_scale=bg_p["fixed_scale"],
        )
        da, _bg_meta = suppressor.suppress(da)
        stages["Background removed"] = da

        # Deferred normalization: apply the user's chosen method now that the
        # subtraction has happened in raw intensity space.
        if defer_norm and (params.get("norm1_enabled") or params.get("norm2_enabled")
                           or params.get("norm3_enabled")):
            norm_p = params.get("norm1") or params.get("norm2") or params.get("norm3") or {}
            if norm_p.get("method"):
                da = _apply_normalize(da, norm_p)
                stages["Normalized (post-suppression)"] = da

    return stages, da


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

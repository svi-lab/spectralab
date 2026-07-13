# -*- coding: utf-8 -*-
"""Preprocessing page: pipeline parameter controls and staged/final spectrum charts."""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
# Background suppression is currently disabled app-wide — these only feed
# _resolve_reference / _render_physics_scale / render_bg_params below, all
# commented out. Kept importable for a future re-enable.
# from backend.background._presets import (
#     list_materials,
#     list_temperatures,
#     list_thicknesses,
#     load_preset,
# )
from ..charts import convert_x, make_comparison_echarts, make_final_echarts, make_progress_echarts
import numpy as np

from ..controls import (
    X_UNIT_FMT,
    X_UNIT_OPTIONS,
    UNIT_DEFAULT,
    render_axis_controls,
    # render_bg_params,  # background suppression disabled app-wide
    render_clean_data_params,
    render_crr_params,
    render_denoising_params,
)
from ..pipeline_cache import final_da, get_finals, stage_dict


# Raw bg widget keys snapshotted to st.session_state["sl_bg_ui"] every rerun —
# background suppression is currently disabled app-wide, so nothing reads
# this. Kept for a future re-enable (see _restore_widget_state below).
# _BG_WIDGET_KEYS = (
#     "bg_ref_source", "bg_ref_file", "bg_row_min", "bg_row_max",
#     "bg_col_min", "bg_col_max",
#     "bg_preset_material", "bg_preset_thickness", "bg_preset_temp",
#     "bg_pt_ratio", "bg_c_override_on", "bg_c_override",
# )


# ---------------------------------------------------------------------------
# Widget state restoration (st.navigation clears widget keys on page switch)
# ---------------------------------------------------------------------------

def _restore_widget_state() -> None:
    """Re-seed widget keys from sl_pipeline_params after navigating back to this page.

    st.navigation clears main-content widget state on page transitions.
    sl_pipeline_params is a plain session-state key, so it survives. We
    pre-seed the widget keys before any widget renders so they pick up the
    saved values rather than their defaults.
    """
    stored = st.session_state.get("sl_pipeline_params")
    if not stored:
        return
    ss = st.session_state

    # ── Normalization ─────────────────────────────────────────────────────────
    if "norm_selection" not in ss:
        sel: list[str] = []
        if stored.get("norm1_enabled"): sel.append("Before")
        if stored.get("norm2_enabled"): sel.append("After CRR")
        if stored.get("norm3_enabled"): sel.append("After Denoising")
        ss["norm_selection"] = sel

    if "norm_method" not in ss:
        method = (
            stored.get("norm1") or stored.get("norm2") or stored.get("norm3") or {}
        ).get("method")
        if method:
            ss["norm_method"] = method

    # ── Clean Data ────────────────────────────────────────────────────────────
    if "cd_enabled" not in ss:
        ss["cd_enabled"] = stored.get("cd_enabled", False)
    cd = stored.get("cd") or {}
    if "cd_n_zeros" not in ss and cd.get("n_zeros") is not None:
        ss["cd_n_zeros"] = cd["n_zeros"]

    # ── Cosmic Ray Remover ────────────────────────────────────────────────────
    if "crr_enabled" not in ss:
        ss["crr_enabled"] = stored.get("crr_enabled", False)
    crr = stored.get("crr") or {}
    if crr:
        if "crr_engine_mode" not in ss:
            ss["crr_engine_mode"] = (
                "1D — per spectrum" if crr.get("force_1d", True)
                else "2D / 3D — collection & spatial"
            )
        for wkey, pkey in (
            ("crr_spike_width",      "spike_width"),
            ("crr_spike_threshold",  "spike_threshold"),
            ("crr_spike_passes",     "spike_passes"),
            ("crr_map_sensitivity",  "map_sensitivity"),
            ("crr_map_disk_radius",  "map_disk_radius"),
            ("crr_map_spike_width",  "map_spike_width"),
            ("crr_map_method",       "map_method"),
            ("crr_map_n_components", "map_n_components"),
        ):
            if wkey not in ss and pkey in crr:
                ss[wkey] = crr[pkey]

    # Background suppression is currently disabled app-wide — no bg widgets
    # are rendered, so there is nothing to restore. Kept for a future
    # re-enable.
#     # ── Background Suppression ────────────────────────────────────────────────
#     if "bg_enabled" not in ss:
#         ss["bg_enabled"] = stored.get("bg_enabled", False)
#     bg_ui = ss.get("sl_bg_ui") or {}
#     loaded_names = set(ss.get("sl_loaded") or {})
#     first_entry = next(iter((ss.get("sl_loaded") or {}).values()), None)
#     target_is_map = bool(first_entry and first_entry["dataset"].is_map)
#     for key, val in bg_ui.items():
#         if key in ss:
#             continue
#         # Skip values that would no longer be a valid widget option.
#         if key == "bg_ref_file" and val not in loaded_names:
#             continue
#         if key == "bg_ref_source" and "Map region" in str(val) and not target_is_map:
#             continue
#         if key == "bg_preset_material" and val not in list_materials():
#             continue
#         if key == "bg_preset_thickness":
#             material = bg_ui.get("bg_preset_material") or ss.get("bg_preset_material")
#             if material and val not in list_thicknesses(material):
#                 continue
#         if key == "bg_preset_temp":
#             material = bg_ui.get("bg_preset_material") or ss.get("bg_preset_material")
#             thickness = bg_ui.get("bg_preset_thickness") or ss.get("bg_preset_thickness")
#             thickness_mm = int(thickness) if material == "glass" and thickness is not None else None
#             if material and val not in list_temperatures(material, thickness_mm):
#                 continue
#         ss[key] = val

    # ── Denoiser ──────────────────────────────────────────────────────────────
    if "denoise_enabled" not in ss:
        ss["denoise_enabled"] = stored.get("denoise_enabled", False)
    den = stored.get("denoise") or {}
    if den:
        if "denoise_engine" not in ss:
            ss["denoise_engine"] = (
                "Smoother — per spectrum" if den.get("per_spectrum")
                else "PCA — population-based"
            )
        for wkey, pkey in (
            ("denoise_nc_type",  "n_components_type"),
            ("denoise_nc_int",   "n_components_int"),
            ("denoise_nc_float", "n_components_float"),
        ):
            if wkey not in ss and pkey in den:
                ss[wkey] = den[pkey]
        if "denoise_baseline" not in ss:
            sub, rst = den.get("subtract_min", True), den.get("restore_min", False)
            ss["denoise_baseline"] = "preserve" if (sub and rst) else ("shape" if sub else "raw")
        sm = den.get("smoother") or {}
        if sm:
            if "denoise_sm_method" not in ss:
                ss["denoise_sm_method"] = sm.get("method", "savgol")
            for wkey, pkey in (
                ("denoise_sm_window_length",   "window_length"),
                ("denoise_sm_polyorder",       "polyorder"),
                ("denoise_sm_d",               "d"),
                ("denoise_sm_auto_lam_calls",  "auto_lam_calls"),
                ("denoise_sm_wavelet",         "wavelet"),
                ("denoise_sm_wavelet_threshold", "wavelet_threshold"),
            ):
                if wkey not in ss and sm.get(pkey) is not None:
                    ss[wkey] = sm[pkey]
            if "denoise_sm_auto_lam" not in ss:
                ss["denoise_sm_auto_lam"] = sm.get("lam") is None
            if "denoise_sm_lam" not in ss and sm.get("lam") is not None:
                ss["denoise_sm_lam"] = sm["lam"]
            if "denoise_sm_wavelet_level" not in ss:
                lv = sm.get("wavelet_level")
                ss["denoise_sm_wavelet_level"] = 0 if lv is None else lv


# ---------------------------------------------------------------------------
# Left column: pipeline parameter widgets
# ---------------------------------------------------------------------------

# Background suppression is currently disabled app-wide — _resolve_reference
# and _render_physics_scale below are unused (only called from the commented
# Background Suppression block in _render_preprocessing_params). Kept for a
# future re-enable.
# def _resolve_reference(
#     bg_params: dict,
#     loaded: dict,
#     ds_name: str,
#     preprocessing_params: dict,
# ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
#     """Resolve the 1-D reference from bg_params["ref_params"].
# 
#     preprocessing_params must have bg_enabled=False — it describes all pipeline
#     stages *except* background suppression, so the reference gets the same CRR
#     / denoising treatment as the main data before the subtraction.
# 
#     Returns (ref_y, ref_x, error_message).
#     ref_x is None when ref and data share the same spectral axis (map region).
#     """
#     ref_p = bg_params.get("ref_params", {})
#     source = ref_p.get("source", "")
# 
#     if source == "Loaded file (mean)":
#         ref_name = ref_p.get("ref_file")
#         if not ref_name or ref_name not in loaded:
#             return None, None, "Reference file not found in loaded files."
# 
#         # Process the reference file through the same pipeline stages as the main
#         # data (CRR, denoising, etc.) so we're subtracting like from like.
#         ref_datasets, ref_errors = get_finals({ref_name: loaded[ref_name]}, preprocessing_params)
#         if ref_errors:
#             return None, None, f"Failed to preprocess reference: {ref_errors[0]}"
# 
#         ref_da = final_da(ref_datasets[ref_name])
#         spectral_dim = loaded[ref_name]["dataset"].spectral_dim
#         ref_x = ref_da.coords[spectral_dim].values
#         non_spectral = [d for d in ref_da.dims if d != spectral_dim]
#         ref_y = np.nanmean(ref_da.values, axis=tuple(range(len(non_spectral)))) if non_spectral else ref_da.values.copy()
# 
#         if np.all(np.isnan(ref_y)):
#             return None, None, "Reference file has all-NaN spectra after preprocessing."
#         return ref_y, ref_x, None
# 
#     elif "Map region" in source:
#         ds = loaded[ds_name]["dataset"]
#         if not ds.is_map:
#             return None, None, "Map region reference requires a map dataset."
#         row_min = ref_p.get("row_min", 0)
#         row_max = ref_p.get("row_max", 0)
#         col_min = ref_p.get("col_min", 0)
#         col_max = ref_p.get("col_max", 0)
# 
#         # Extract the region from the preprocessed map (same stages as main data).
#         main_datasets, main_errors = get_finals({ds_name: loaded[ds_name]}, preprocessing_params)
#         if main_errors:
#             return None, None, f"Failed to preprocess map for region extraction: {main_errors[0]}"
# 
#         da = final_da(main_datasets[ds_name])
#         try:
#             region = da.isel({da.dims[0]: slice(row_min, row_max + 1),
#                               da.dims[1]: slice(col_min, col_max + 1)})
#         except Exception as e:
#             return None, None, f"Invalid map region: {e}"
#         ref_y = np.nanmean(region.values.reshape(-1, region.shape[-1]), axis=0)
#         if np.all(np.isnan(ref_y)):
#             return None, None, "Selected map region is all-NaN."
#         return ref_y, None, None
# 
#     elif source == "Built-in preset":
#         preset_key = ref_p.get("preset_key")
#         if not preset_key:
#             return None, None, "No built-in preset selected."
#         try:
#             ref_x, ref_y, _meta = load_preset(preset_key)
#         except (FileNotFoundError, KeyError) as e:
#             return None, None, f"Failed to load preset: {e}"
#         if np.all(np.isnan(ref_y)):
#             return None, None, "Selected preset is all-NaN."
#         return ref_y, ref_x, None
# 
#     return None, None, "Unknown reference source."


# def _render_physics_scale(bg_params_raw: dict, loaded: dict, ds_name: str) -> float | None:
#     """Resolve the physics-mode scale c_total = c_physics × (P·t)_sample/(P·t)_ref.
# 
#     c_physics is read from the Data page's Sample Structure summary (the single
#     source of truth) — no optics calls happen here. Renders the factor
#     breakdown, a manual ratio input when power/exposure metadata is missing,
#     and a manual override for the final c. Returns None when the structure is
#     missing or the file is marked as a bare substrate — the caller disables
#     the stage in that case.
#     """
#     ss_info = st.session_state.get("sl_sample_structure", {}).get(ds_name)
#     if ss_info and ss_info.get("sample_type") == "substrate":
#         st.error(
#             f"**{ds_name}** is marked as a bare substrate on the Data page — "
#             "there is no film to suppress under. Fix the sample type or "
#             "process the film file instead."
#         )
#         return None
#     summary = (ss_info or {}).get("summary") or {}
#     if "c_physics" not in summary:
#         st.error(
#             "Physics mode needs the film and substrate parameters — "
#             "fill in **Sample Structure** on the Data page."
#         )
#         return None
# 
#     c_optical = float(summary["c_physics"])
# 
#     # ── Power × exposure ratio between sample and reference ──────────────
#     source = bg_params_raw.get("ref_params", {}).get("source", "")
#     if "Map region" in source:
#         ratio = 1.0
#         ratio_note = "same file → ratio = 1"
#     elif source == "Built-in preset":
#         preset_key = bg_params_raw.get("ref_params", {}).get("preset_key")
#         tgt = loaded[ds_name]["dataset"]
#         pt_y = tgt.laser_power * tgt.exposure_time
#         try:
#             _ref_x, _ref_y, preset_meta = load_preset(preset_key)
#             pt_r = preset_meta["laser_power"] * preset_meta["exposure_time"]
#         except (FileNotFoundError, KeyError, TypeError):
#             pt_r = float("nan")
#         if np.isfinite(pt_y) and np.isfinite(pt_r) and pt_r > 0:
#             ratio = pt_y / pt_r
#             ratio_note = (
#                 f"(P·t)ₛ {tgt.laser_power:g}×{tgt.exposure_time:g} / "
#                 f"(P·t)preset {preset_meta['laser_power']:g}×{preset_meta['exposure_time']:g}"
#             )
#         else:
#             ratio = float(st.number_input(
#                 "Power × exposure ratio (sample / reference)",
#                 value=1.0, min_value=0.0, step=0.1, format="%.3f",
#                 key="bg_pt_ratio",
#                 help=(
#                     "Laser power or exposure time is missing from the file "
#                     "metadata — enter (P_sample·t_sample)/(P_ref·t_ref) manually. "
#                     "1.0 = identical acquisition conditions."
#                 ),
#             ))
#             ratio_note = "metadata missing — manual"
#     else:
#         tgt = loaded[ds_name]["dataset"]
#         ref_name = bg_params_raw.get("ref_params", {}).get("ref_file")
#         rds = loaded[ref_name]["dataset"] if ref_name in loaded else None
#         pt_y = tgt.laser_power * tgt.exposure_time
#         pt_r = (rds.laser_power * rds.exposure_time) if rds is not None else float("nan")
#         if np.isfinite(pt_y) and np.isfinite(pt_r) and pt_r > 0:
#             ratio = pt_y / pt_r
#             ratio_note = (
#                 f"(P·t)ₛ {tgt.laser_power:g}×{tgt.exposure_time:g} / "
#                 f"(P·t)ᵣ {rds.laser_power:g}×{rds.exposure_time:g}"
#             )
#         else:
#             ratio = float(st.number_input(
#                 "Power × exposure ratio (sample / reference)",
#                 value=1.0, min_value=0.0, step=0.1, format="%.3f",
#                 key="bg_pt_ratio",
#                 help=(
#                     "Laser power or exposure time is missing from the file "
#                     "metadata — enter (P_sample·t_sample)/(P_ref·t_ref) manually. "
#                     "1.0 = identical acquisition conditions."
#                 ),
#             ))
#             ratio_note = "metadata missing — manual"
# 
#     c_total = c_optical * ratio
#     st.caption(
#         f"c = T→sub {summary['tmm_T_sub']:.3f} / (1−R_ref) {1.0 - summary['R_air_sub']:.3f} "
#         f"× power {ratio:.3f} ({ratio_note}) "
#         f"= **{c_total:.4f}**. "
#         "Assumes PL scales linearly with excitation power."
#     )
# 
#     # ── Manual override ───────────────────────────────────────────────────
#     if st.checkbox("Override c manually", key="bg_c_override_on"):
#         c_total = float(st.number_input(
#             "Scale c", min_value=0.0, value=float(c_total), step=0.01,
#             format="%.4f", key="bg_c_override",
#         ))
#     return c_total


def _render_preprocessing_params(processing_ok: bool, loaded: dict, ds_name: str) -> dict[str, Any]:
    """Render pipeline parameter widgets; return assembled pipeline_params."""

    # ── Normalization (always visible, above the processing tabs) ─────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">Normalization</p>', unsafe_allow_html=True)
        _NORM_SEGMENTS = ["Before", "After CRR", "After Denoising"]
        norm_selection = st.segmented_control(
            "Normalize at",
            _NORM_SEGMENTS,
            selection_mode="multi",
            key="norm_selection",
            label_visibility="collapsed",
        )
        norm_method: str | None = None
        if norm_selection:
            norm_method = st.selectbox(
                "Method",
                ["min_max", "area"],
                key="norm_method",
                format_func=lambda m: {"min_max": "Min-Max", "area": "Area"}[m],
                help=(
                    "**Min-Max** — shifts and scales each spectrum so its minimum "
                    "becomes 0 and its maximum becomes 1.  Fast and shape-preserving; "
                    "good for comparing peak positions and relative heights when "
                    "absolute intensity differences do not matter.  Sensitive to "
                    "outlier spikes: a single very high or very low point will "
                    "compress the rest of the spectrum.\n\n"
                    "**Area** — divides each spectrum by its trapezoidal integral "
                    "(area under the curve), then shifts the floor to 0.  Preserves "
                    "the relative weight of broad vs. narrow features and is robust "
                    "to isolated spikes.  Use this when you want spectra that "
                    "represent the same total 'amount' of signal — e.g. before "
                    "comparing integrated intensities across samples."
                ),
            )

    # ── Clean Data ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">Clean Data</p>', unsafe_allow_html=True)
        cd_enabled = st.toggle(
            "Remove oversaturated spectra",
            key="cd_enabled",
            help=(
                "Scans every spectrum for **ADC saturation artefacts** — consecutive "
                "channels stuck at exactly 0, which occur when the detector clips to "
                "zero instead of recording the true signal.\n\n"
                "**What it does by data shape:**\n"
                "- **Single spectrum (1D):** issues a warning; spectrum is left unchanged.\n"
                "- **Line scan / series (2D):** drops saturated spectra from the stack "
                "and records which indices were removed.\n"
                "- **Map (3D):** NaN-fills the dead pixels in place, preserving the full "
                "map shape so spatial coordinates stay intact. All downstream steps "
                "(Cosmic Ray Removal, Spectra Cleaner) handle NaN pixels gracefully.\n\n"
                "**When to enable:** if your data contains dead detector pixels or "
                "spectra where the signal went off-scale and clipped to zero. "
                "Run this *before* Cosmic Ray Removal so dead pixels don't interfere "
                "with the spatial reference computation.\n\n"
                "**n_zeros threshold:** how many consecutive zero-valued channels "
                "define a saturated spectrum. The default of 10 avoids false positives "
                "from small gaps while catching real saturation events."
            ),
        )
        cd_params: dict[str, Any] = {}
        if cd_enabled:
            cd_params = render_clean_data_params()

    # ── CosmicRayRemover ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">Cosmic Ray Remover</p>', unsafe_allow_html=True)
        if not processing_ok:
            st.info(
                "Requires PL data (Nanometer or ElectronVolt). "
                "Not available for this upload."
            )
        crr_enabled = st.toggle(
            "Apply CosmicRayRemover", key="crr_enabled", disabled=not processing_ok
        )
        crr_params: dict[str, Any] = {}
        if crr_enabled:
            crr_params = render_crr_params()

    # ── Denoising ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">Denoising</p>', unsafe_allow_html=True)
        if not processing_ok:
            st.info(
                "Requires PL data (Nanometer or ElectronVolt). "
                "Not available for this upload."
            )
        denoise_enabled = st.toggle(
            "Apply Denoiser", key="denoise_enabled", disabled=not processing_ok
        )
        denoise_params: dict[str, Any] = {}
        if denoise_enabled:
            denoise_params = render_denoising_params()

    # ── Background Suppression ────────────────────────────────────────────────
    bg_enabled = False
    bg_params_raw: dict[str, Any] = {}
    bg_pipeline: dict[str, Any] = {}

    # Background suppression is currently disabled app-wide — no controls
    # are rendered, so bg_enabled/bg_params_raw/bg_pipeline stay at the
    # defaults above. Kept for a future re-enable.
#     # Pipeline params for all stages above (used to preprocess the reference
#     # with the same CRR/denoising before the subtraction).
#     _ns_pre = norm_selection or []
#     _nm_pre = {"method": norm_method} if norm_method else {}
#     _intermediate_params: dict[str, Any] = {
#         "norm1_enabled":   "Before" in _ns_pre,        "norm1":   _nm_pre,
#         "cd_enabled":      cd_enabled,                 "cd":      cd_params,
#         "crr_enabled":     crr_enabled,                "crr":     crr_params,
#         "norm2_enabled":   "After CRR" in _ns_pre,     "norm2":   _nm_pre,
#         "denoise_enabled": denoise_enabled,            "denoise": denoise_params,
#         "norm3_enabled":   "After Denoising" in _ns_pre, "norm3": _nm_pre,
#         "bg_enabled":      False,                      "bg":      {},
#     }
# 
#     with st.container(border=True):
#         st.markdown('<p class="section-header">Background Suppression</p>', unsafe_allow_html=True)
#         bg_enabled = st.toggle(
#             "Subtract substrate reference",
#             key="bg_enabled",
#             help=(
#                 "Subtracts a scaled bare-substrate spectrum from each spectrum: "
#                 "**corrected = measured − c · reference**.\n\n"
#                 "Scale factor c comes from the optical model (Sample Structure "
#                 "on the Data page), scaled by the laser-power × exposure ratio "
#                 "between sample and reference. Normalization is applied "
#                 "*after* the subtraction."
#             ),
#         )
#         if bg_enabled:
#             ref_ds = loaded[ds_name]["dataset"]
#             bg_params_raw = render_bg_params(loaded, ds_name, ref_ds.is_map)
# 
#             if not bg_params_raw.get("valid", False):
#                 bg_enabled = False
#             else:
# 
#                 # Suppression always runs in un-normalized space,
#                 # so the reference skips the normalization stages too — the
#                 # pipeline defers them until after the subtraction.
#                 ref_pre_params = {
#                     **_intermediate_params,
#                     "norm1_enabled": False, "norm2_enabled": False, "norm3_enabled": False,
#                 }
# 
#                 ref_y, ref_x, ref_err = _resolve_reference(bg_params_raw, loaded, ds_name, ref_pre_params)
#                 if ref_err:
#                     st.error(f"Reference error: {ref_err}")
#                     bg_enabled = False
#                 else:
#                     if norm_selection:
#                         st.info(
#                             "Your normalization runs **after** the subtraction — "
#                             "the suppression always operates on un-normalized "
#                             "spectra. See the 'Normalized (post-suppression)' "
#                             "stage in the Progress tab."
#                         )
# 
#                     c_total = _render_physics_scale(bg_params_raw, loaded, ds_name)
#                     if c_total is None:
#                         bg_enabled = False
#                     else:
#                         bg_pipeline = {
#                             "reference":   ref_y,
#                             "reference_x": ref_x,
#                             "scale_mode":  "physics",
#                             "fixed_scale": float(c_total),
#                         }
# 
#         # Snapshot raw bg widget values for cross-page persistence (see
#         # _BG_WIDGET_KEYS / _restore_widget_state).
#         _bg_ui_prev = st.session_state.get("sl_bg_ui") or {}
#         st.session_state["sl_bg_ui"] = {
#             **_bg_ui_prev,
#             **{k: st.session_state[k] for k in _BG_WIDGET_KEYS if k in st.session_state},
#         }

    # ── Assemble and return pipeline_params ───────────────────────────────────
    _ns = norm_selection or []
    _nm = {"method": norm_method} if norm_method else {}
    return {
        "norm1_enabled":   "Before" in _ns,
        "norm1":           _nm,
        "cd_enabled":      cd_enabled,
        "cd":              cd_params,
        "crr_enabled":     crr_enabled,
        "crr":             crr_params,
        "norm2_enabled":   "After CRR" in _ns,
        "norm2":           _nm,
        "denoise_enabled": denoise_enabled,
        "denoise":         denoise_params,
        "norm3_enabled":   "After Denoising" in _ns,
        "norm3":           _nm,
        "bg_enabled":      bg_enabled,
        "bg":              bg_pipeline,
    }


# ---------------------------------------------------------------------------
# Right column: preprocessing execution and chart tabs
# ---------------------------------------------------------------------------

def _run_preprocessing(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
) -> tuple[dict, list[str]]:
    with st.spinner(f"Processing {len(loaded)} file(s)…"):
        # keep_stages: the Progress tab is the one consumer of intermediates.
        return get_finals(loaded, pipeline_params, keep_stages=True)


@st.cache_data(show_spinner=False, max_entries=16)
def _make_final_echarts_cached(
    file_hash: str,
    pipeline_params: dict,
    _da,
    title: str,
    color_by: str,
    x_unit: str,
    laser_nm: float | None,
    src_unit: str,
    native_type: str,
) -> dict:
    return make_final_echarts(
        _da, title=title, color_by=color_by,
        x_unit=x_unit, laser_nm=laser_nm, src_unit=src_unit, native_type=native_type,
    )


def _render_progress_tab(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> None:
    default_unit = UNIT_DEFAULT.get(ref_ds.spectral_units, "wavelength")
    current_unit = st.session_state.get("prog_x_unit", default_unit)

    x_native = ref_ds.da.coords[ref_ds.spectral_dim].values
    x_disp = convert_x(
        x_native, ref_ds.spectral_dim, current_unit,
        ref_ds.laser_nm, src_unit=ref_ds.spectral_unit,
        native_type=ref_ds.spectral_units,
    )
    disp_min = float(x_disp.min())
    disp_max = float(x_disp.max())

    col_unit, col_from, col_to, col_laser, col_title = st.columns([2, 1, 1, 1, 2])

    with col_unit:
        x_unit = st.selectbox(
            "Spectral units", X_UNIT_OPTIONS,
            format_func=X_UNIT_FMT.get,
            index=X_UNIT_OPTIONS.index(current_unit),
            key="prog_x_unit",
        )

    with col_from:
        _x_step = 0.01 if x_unit == "energy" else 1.0
        _x_fmt = "%.2f" if x_unit == "energy" else "%.0f"
        x_from = st.number_input(
            "From", value=min(disp_min, disp_max),
            key=f"prog_from_{x_unit}", format=_x_fmt, step=_x_step,
        )

    with col_to:
        x_to = st.number_input(
            "To", value=max(disp_min, disp_max),
            key=f"prog_to_{x_unit}", format=_x_fmt, step=_x_step,
        )

    laser = ref_ds.laser_nm
    with col_laser:
        if x_unit == "raman_shift" and laser is None:
            laser = st.number_input(
                "Laser (nm)", value=532.0, min_value=1.0, step=0.1,
                key="prog_laser_nm",
                help="Not found in file — enter the excitation wavelength.",
            )

    x_range = (min(x_from, x_to), max(x_from, x_to))

    default_title = next(iter(all_datasets)) if not multi else "Final spectra — all files"
    with col_title:
        chart_title = st.text_input("Chart title", value=default_title, key="prog_title")

    if multi:
        finals = {name: final_da(ds) for name, ds in all_datasets.items()}
        opts = make_comparison_echarts(
            finals, title=chart_title,
            x_unit=x_unit, laser_nm=laser,
            src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
            x_range=x_range,
        )
    else:
        name = next(iter(all_datasets))
        opts = make_progress_echarts(
            stage_dict(all_datasets[name]), title=chart_title,
            x_unit=x_unit, laser_nm=laser,
            src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
            x_range=x_range,
        )

    st_echarts(opts, height="72vh", key="progress_chart")


def _render_final_tab(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> None:
    if multi:
        view_mode = st.radio(
            "View", ["Comparison (all files)", "Single file"],
            horizontal=True, key="final_view_mode",
        )
    else:
        view_mode = "Single file"

    if view_mode == "Comparison (all files)":
        chart_title = st.text_input("Chart title", value="Comparison — all files", key="fin_cmp_title")
        x_unit, laser = render_axis_controls(
            "fin_cmp",
            ref_ds.laser_nm,
            native_type=ref_ds.spectral_units,
        )
        finals = {name: final_da(ds) for name, ds in all_datasets.items()}
        st_echarts(
            make_comparison_echarts(
                finals, title=chart_title,
                x_unit=x_unit, laser_nm=laser,
                src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
            ),
            height="72vh", key="final_comparison",
        )

    else:
        if multi:
            selected = st.selectbox(
                "Select file", list(all_datasets.keys()), key="final_file_select"
            )
        else:
            selected = next(iter(all_datasets))

        sel_ds: SpectralDataset = loaded[selected]["dataset"]
        da_sel = final_da(all_datasets[selected])

        n_spectra = int(da_sel.size // da_sel.shape[-1]) if da_sel.ndim > 1 else 1
        if n_spectra > 5000:
            st.warning(
                f"Large dataset ({n_spectra} spectra). "
                "Only a subset of spectra is drawn in index mode "
                "(display is also downsampled to ~1,200 points/spectrum along "
                "the spectral axis). Exports and analysis stay full-resolution.",
                icon="⚠️",
            )

        ctl1, ctl2 = st.columns([2, 2])
        color_by = ctl1.selectbox(
            "Color mode",
            ["index", "mean_dev"],
            format_func=lambda x: {
                "index":    "Index (spectrum order)",
                "mean_dev": "Mean deviation",
            }[x],
            key="final_color_by",
        )
        chart_title = ctl2.text_input("Chart title", value=selected, key="fin_single_title")

        x_unit, laser = render_axis_controls(
            "fin_single",
            sel_ds.laser_nm,
            native_type=sel_ds.spectral_units,
        )
        st_echarts(
            _make_final_echarts_cached(
                loaded[selected]["hash"], pipeline_params, da_sel, chart_title,
                color_by,
                x_unit, laser, sel_ds.spectral_unit, sel_ds.spectral_units,
            ),
            height="72vh", key="final_single",
        )


@st.fragment
def _render_charts_fragment(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> None:
    """Fragment that owns both chart tabs.

    Creating st.tabs() inside the fragment (rather than passing tab objects
    in from outside) is required by Streamlit's fragment contract: fragments
    may only render into containers they create themselves.
    """
    tab_prog, tab_final = st.tabs(["Preprocessing", "Final"])
    with tab_prog:
        _render_progress_tab(all_datasets, loaded, pipeline_params, multi, ref_ds)
    with tab_final:
        _render_final_tab(all_datasets, loaded, pipeline_params, multi, ref_ds)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_preprocessing_page() -> None:
    """Preprocessing page: pipeline controls (left) + staged and final charts (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    _restore_widget_state()

    processing_ok: bool = st.session_state.get("sl_processing_ok", False)
    ref_ds: SpectralDataset = next(iter(loaded.values()))["dataset"]

    left, right = st.columns([1, 2], gap="medium")

    # Use first file as the "target" for bg reference resolution.
    # When multiple files are loaded the reference selectbox naturally excludes
    # the target file, so this is the right anchor for single-file use.
    target_name = next(iter(loaded))

    with left:
        pipeline_params = _render_preprocessing_params(processing_ok, loaded, target_name)
        st.session_state["sl_pipeline_params"] = pipeline_params

    with right:
        all_datasets, errors = _run_preprocessing(loaded, pipeline_params)

        for err in errors:
            st.error(f"Processing error — {err}")
        if not all_datasets:
            st.stop()

        multi = len(all_datasets) > 1

        _render_charts_fragment(all_datasets, loaded, pipeline_params, multi, ref_ds)

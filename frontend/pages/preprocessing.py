# -*- coding: utf-8 -*-
"""Preprocessing page: pipeline parameter controls and staged/final spectrum charts."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
from backend._shared.scan_geometry import ScanGeometry, get_scan_geometry
from backend._shared.scan_overlay import draw_scan_overlay
from ..charts import convert_x, make_comparison_echarts, make_final_echarts, make_progress_echarts
from ..controls import (
    X_UNIT_FMT,
    X_UNIT_OPTIONS,
    UNIT_DEFAULT,
    render_axis_controls,
    render_clean_data_params,
    render_crr_params,
    render_denoising_params,
)
from ..pipeline_cache import get_finals


@st.cache_data(show_spinner=False, max_entries=16)
def _draw_overlay_cached(
    file_hash: str,
    pipeline_params: dict,
    _image_arr: np.ndarray,
    image_meta: dict,
    _geo: ScanGeometry,
    _removed_mask: np.ndarray | None,
) -> np.ndarray:
    """Scan-footprint overlay drawing — a per-point PIL loop that's wasted
    work to repeat on reruns triggered by unrelated chart controls."""
    return draw_scan_overlay(_image_arr, image_meta, _geo, removed_mask=_removed_mask)


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
# Image helpers
# ---------------------------------------------------------------------------

def _show_images(
    names: list[str],
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    all_finals: dict[str, Any] | None = None,
) -> None:
    imgs = [
        (name, loaded[name]["dataset"])
        for name in names
        if loaded.get(name) and loaded[name]["dataset"].image_arr is not None
    ]
    if not imgs:
        return
    st.divider()
    _, img_col, _ = st.columns([1, 8, 1])
    with img_col:
        n_per_row = min(len(imgs), 4)
        for i in range(0, len(imgs), n_per_row):
            batch = imgs[i: i + n_per_row]
            cols = st.columns(len(batch))
            for col, (name, ds) in zip(cols, batch):
                col.markdown(f"**{name}**")
                arr = ds.image_arr
                geo = get_scan_geometry(ds)
                if geo is not None and ds.image_meta is not None:
                    removed_mask = _get_removed_mask(geo, ds, all_finals, name)
                    arr = _draw_overlay_cached(
                        loaded[name]["hash"], pipeline_params,
                        arr, ds.image_meta, geo, removed_mask,
                    )
                col.image(arr, width="stretch")


def _get_removed_mask(geo, ds, all_finals, name) -> np.ndarray | None:
    """Return bool mask (len = len(geo.xs)) marking CleanData-removed points."""
    if geo.shape != "points" or all_finals is None:
        return None
    da_final = all_finals.get(name)
    if da_final is None:
        return None
    if da_final.ndim == 3:
        nan_mask = np.isnan(da_final.values).all(axis=-1)
        if not nan_mask.any():
            return None
        return nan_mask.ravel()
    if da_final.ndim == 2 and "point" in da_final.coords and "point" in ds.da.coords:
        final_ids = set(da_final.coords["point"].values.tolist())
        mask = np.array([
            pid not in final_ids
            for pid in ds.da.coords["point"].values.tolist()
        ])
        return mask if mask.any() else None
    return None


# ---------------------------------------------------------------------------
# Left column: pipeline parameter widgets
# ---------------------------------------------------------------------------

def _render_preprocessing_params(processing_ok: bool) -> dict[str, Any]:
    """Render all pipeline parameter widgets; return assembled pipeline_params."""

    # ── Normalization ─────────────────────────────────────────────────────────
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

    st.divider()

    # ── Clean Data ────────────────────────────────────────────────────────────
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

    st.divider()

    # ── CosmicRayRemover ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Cosmic Ray Remover</p>', unsafe_allow_html=True)
    if not processing_ok:
        st.info(
            "CosmicRayRemover and Denoiser require PL data "
            "(Nanometer or ElectronVolt). Not available for this upload."
        )
    crr_enabled = st.toggle(
        "Apply CosmicRayRemover", key="crr_enabled", disabled=not processing_ok
    )
    crr_params: dict[str, Any] = {}
    if crr_enabled:
        crr_params = render_crr_params()

    st.divider()

    # ── Denoising ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Denoising</p>', unsafe_allow_html=True)
    denoise_enabled = st.toggle(
        "Apply Denoiser", key="denoise_enabled", disabled=not processing_ok
    )
    denoise_params: dict[str, Any] = {}
    if denoise_enabled:
        denoise_params = render_denoising_params()

    st.divider()

    # ── Background Suppression (placeholder) ──────────────────────────────────
    st.markdown('<p class="section-header">Background Suppression</p>', unsafe_allow_html=True)
    st.info("Coming soon — background suppression controls will appear here.")

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
    }


# ---------------------------------------------------------------------------
# Right column: preprocessing execution and chart tabs
# ---------------------------------------------------------------------------

def _run_preprocessing(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
) -> tuple[dict, dict, list[str]]:
    with st.spinner(f"Processing {len(loaded)} file(s)…"):
        return get_finals(loaded, pipeline_params)


@st.cache_data(show_spinner=False, max_entries=16)
def _make_final_echarts_cached(
    file_hash: str,
    pipeline_params: dict,
    _da,
    title: str,
    color_by: str,
    n_bins: int | None,
    step: int,
    x_unit: str,
    laser_nm: float | None,
    src_unit: str,
    native_type: str,
) -> dict:
    """The density/density_lines modes rebin the whole spectra array — worth
    skipping on reruns where only an unrelated control on the page changed."""
    return make_final_echarts(
        _da, title=title, color_by=color_by, n_bins=n_bins, step=step,
        x_unit=x_unit, laser_nm=laser_nm, src_unit=src_unit, native_type=native_type,
    )


@st.fragment
def _render_progress_tab(
    tab,
    all_stages: dict,
    all_finals: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> None:
    with tab:
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
            x_from = st.number_input(
                "From", value=min(disp_min, disp_max),
                key=f"prog_from_{x_unit}", format="%.2f",
            )

        with col_to:
            x_to = st.number_input(
                "To", value=max(disp_min, disp_max),
                key=f"prog_to_{x_unit}", format="%.2f",
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

        default_title = next(iter(all_stages)) if not multi else "Final spectra — all files"
        with col_title:
            chart_title = st.text_input("Chart title", value=default_title, key="prog_title")

        if multi:
            opts = make_comparison_echarts(
                all_finals, title=chart_title,
                x_unit=x_unit, laser_nm=laser,
                src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
                x_range=x_range,
            )
            names_img = list(loaded.keys())
        else:
            name = next(iter(all_stages))
            opts = make_progress_echarts(
                all_stages[name], title=chart_title,
                x_unit=x_unit, laser_nm=laser,
                src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
                x_range=x_range,
            )
            names_img = [name]

        st_echarts(opts, height="72vh", key="progress_chart")
        _show_images(names_img, loaded, pipeline_params, all_finals)


@st.fragment
def _render_final_tab(
    tab,
    all_finals: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> None:
    with tab:
        if multi:
            view_mode = st.radio(
                "View", ["Comparison (all files)", "Single file"],
                horizontal=True, key="final_view_mode",
            )
        else:
            view_mode = "Single file"

        if view_mode == "Comparison (all files)":
            x_unit, laser = render_axis_controls(
                "fin_cmp",
                ref_ds.laser_nm,
                native_type=ref_ds.spectral_units,
            )
            st_echarts(
                make_comparison_echarts(
                    all_finals, title="Comparison — final processed",
                    x_unit=x_unit, laser_nm=laser,
                    src_unit=ref_ds.spectral_unit, native_type=ref_ds.spectral_units,
                ),
                height="72vh", key="final_comparison",
            )
            _show_images(list(loaded.keys()), loaded, pipeline_params, all_finals)

        else:
            if multi:
                selected = st.selectbox(
                    "Select file", list(all_finals.keys()), key="final_file_select"
                )
            else:
                selected = next(iter(all_finals))

            sel_ds: SpectralDataset = loaded[selected]["dataset"]
            da_sel = all_finals[selected]

            n_spectra = int(da_sel.size // da_sel.shape[-1]) if da_sel.ndim > 1 else 1
            if n_spectra > 5000:
                st.warning(
                    f"Large dataset ({n_spectra} spectra). "
                    "Consider using 'density' or 'density_lines' mode.",
                    icon="⚠️",
                )

            ctl1, ctl2, ctl3 = st.columns([2, 1, 1])
            color_by = ctl1.selectbox(
                "Color mode",
                ["index", "density", "density_lines", "mean_dev"],
                format_func=lambda x: {
                    "index":         "Index (spectrum order)",
                    "density":       "Density (2D histogram)",
                    "density_lines": "Density lines",
                    "mean_dev":      "Mean deviation",
                }[x],
                key="final_color_by",
            )
            step = ctl2.number_input(
                "x step", value=10, min_value=1, step=1, key="final_step",
                help="Downsample spectral axis by this factor.",
            )
            n_bins_input = ctl3.number_input(
                "n_bins", value=200, min_value=10, max_value=200, step=10, key="final_nbins",
                help="Intensity bins (density modes only). Max 200.",
            )
            n_bins_val = int(n_bins_input) if color_by in ("density", "density_lines") else None

            x_unit, laser = render_axis_controls(
                "fin_single",
                sel_ds.laser_nm,
                native_type=sel_ds.spectral_units,
            )
            st_echarts(
                _make_final_echarts_cached(
                    loaded[selected]["hash"], pipeline_params, da_sel, selected,
                    color_by, n_bins_val, int(step),
                    x_unit, laser, sel_ds.spectral_unit, sel_ds.spectral_units,
                ),
                height="72vh", key="final_single",
            )
            _show_images([selected], loaded, pipeline_params, all_finals)


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

    with left:
        pipeline_params = _render_preprocessing_params(processing_ok)
        st.session_state["sl_pipeline_params"] = pipeline_params

    with right:
        all_stages, all_finals, errors = _run_preprocessing(loaded, pipeline_params)

        for err in errors:
            st.error(f"Processing error — {err}")
        if not all_finals:
            st.stop()

        multi = len(all_finals) > 1

        tab_prog, tab_final = st.tabs(["Preprocessing", "Final"])
        _render_progress_tab(tab_prog, all_stages, all_finals, loaded, pipeline_params, multi, ref_ds)
        _render_final_tab(tab_final, all_finals, loaded, pipeline_params, multi, ref_ds)

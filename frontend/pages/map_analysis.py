# -*- coding: utf-8 -*-
"""Map Analysis page: spectral map parameters and heatmap visualization."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend._shared.dataset import SpectralDataset
from streamlit_echarts import st_echarts

from ..charts import convert_x, convert_x_to_native
from ..controls import render_axis_controls, render_map_display_controls
from ..map_chart import make_map_fig, make_mean_spectrum_option
from ..pipeline_cache import default_pipeline_params, final_da, get_finals

# Short colorbar-caption label for the *display* x_unit (from render_axis_controls),
# not the file's native storage unit — the map/slider always operate in whatever unit
# the user has selected.
_SHORT_UNIT = {
    "wavelength":  "nm",
    "energy":      "eV",
    "wavenumber":  "cm⁻¹",
    "raman_shift": "cm⁻¹",
}


@st.cache_data(show_spinner=False, max_entries=16)
def _make_map_fig_cached(
    file_hash: str,
    pipeline_params: dict,
    _da_map,
    _image_arr,
    image_meta: dict | None,
    lmin: float,
    lmax: float,
    quantity: str,
    colorscale: str,
    title: str,
    spectral_unit: str,
    map_opacity: float,
    label_min: float,
    label_max: float,
):
    """Spectral-range integration/deviation + figure build — recomputed on
    every slider tick otherwise, even when other page widgets trigger the
    rerun instead."""
    return make_map_fig(
        da=_da_map, image_arr=_image_arr, image_meta=image_meta,
        lambda_min=lmin, lambda_max=lmax, quantity=quantity,
        colorscale=colorscale, title=title, spectral_unit=spectral_unit,
        map_opacity=map_opacity, label_min=label_min, label_max=label_max,
    )


@st.fragment
def _render_map_section(
    da_map, ds: SpectralDataset, map_name: str, file_hash: str,
    pipeline_params: dict[str, Any], quantity: str, colorscale: str,
    x_unit: str, laser_nm: float | None, map_opacity: float,
) -> None:
    spectral_dim = da_map.dims[-1]
    x_native = da_map.coords[spectral_dim].values
    x_disp = convert_x(
        x_native, spectral_dim, x_unit, laser_nm,
        src_unit=ds.spectral_unit, native_type=ds.spectral_units,
    )
    # .min()/.max() rather than [0]/[-1]: eV is inversely related to nm, so converting
    # an ascending native coordinate can flip the array to descending.
    disp_min = float(x_disp.min())
    disp_max = float(x_disp.max())
    is_energy = x_unit == "energy"
    step = 0.01 if is_energy else max(round((disp_max - disp_min) / len(x_disp), 2), 0.1)
    ndigits = 2 if is_energy else 1
    _default_lmin = round(disp_min + (disp_max - disp_min) * 0.2, ndigits)
    _default_lmax = round(disp_min + (disp_max - disp_min) * 0.8, ndigits)

    # Per-unit session-state key: switching units mid-session must not hand a stale
    # nm-range value to a slider now bounded in eV (same pattern as preprocessing.py's
    # f"prog_from_{x_unit}"/f"prog_to_{x_unit}").
    range_key = f"map_spec_range_{x_unit}"
    spectral_unit_label = _SHORT_UNIT.get(x_unit, x_unit)

    with st.spinner("Building map…"):
        lmin_disp, lmax_disp = st.session_state.get(range_key, (_default_lmin, _default_lmax))
        if lmin_disp < lmax_disp:
            lmin_native = convert_x_to_native(
                lmin_disp, spectral_dim, x_unit, laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )
            lmax_native = convert_x_to_native(
                lmax_disp, spectral_dim, x_unit, laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )
            fig_map = _make_map_fig_cached(
                file_hash, pipeline_params, da_map, ds.image_arr, ds.image_meta,
                lmin_native, lmax_native, quantity, colorscale, map_name, spectral_unit_label,
                map_opacity, lmin_disp, lmax_disp,
            )
            st.plotly_chart(fig_map, width="stretch", height=600)

    if ds.image_arr is None:
        st.caption("ℹ No white-light image (WHTL block) found in this file — heatmap only.")
    elif ds.image_meta is None:
        st.caption("ℹ Image found but EXIF geo-registration data missing — heatmap only.")

    _, _slider_col, _ = st.columns([0.5, 11, 0.24])  # adjust outer cols to tune width
    with _slider_col:
        lmin_disp, lmax_disp = st.slider(
            "Spectral range",
            min_value=disp_min,
            max_value=disp_max,
            value=(_default_lmin, _default_lmax),
            step=step,
            key=range_key,
            label_visibility="collapsed",
        )
    if lmin_disp >= lmax_disp:
        st.warning("Left handle must be less than right handle.")

    mean_da = da_map.mean([d for d in da_map.dims if d != spectral_dim], skipna=True)
    st_echarts(
        make_mean_spectrum_option(
            mean_da, lmin_disp, lmax_disp, x_unit, laser_nm,
            src_unit=ds.spectral_unit, native_type=ds.spectral_units,
        ),
        height="200px",
    )


def render_map_page() -> None:
    """Map Analysis page: map parameters (left) + plotly heatmap (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    pipeline_params = st.session_state.get("sl_pipeline_params") or default_pipeline_params()
    with st.spinner("Preparing data…"):
        all_datasets, _errors = get_finals(loaded, pipeline_params)

    map_candidates = {
        name: entry for name, entry in loaded.items()
        if entry["dataset"].is_map
    }

    if not map_candidates:
        st.info(
            "No raster map files loaded. "
            "The Map Analysis page works with 3D (row, column, spectral) DataArrays."
        )
        return

    left, right = st.columns([1, 2], gap="medium")

    with left:
        if len(map_candidates) > 1:
            map_name = st.selectbox(
                "Select file", list(map_candidates.keys()), key="map_file_select"
            )
        else:
            map_name = next(iter(map_candidates))

        ds: SpectralDataset = map_candidates[map_name]["dataset"]

        with st.container(border=True):
            st.markdown('<p class="section-header">Display</p>', unsafe_allow_html=True)
            x_unit, laser_nm = render_axis_controls(
                "map", ds.laser_nm, native_type=ds.spectral_units,
            )
            quantity = st.selectbox(
                "Quantity",
                ["integrated", "deviation"],
                format_func=lambda q: {
                    "integrated": "Integrated intensity",
                    "deviation":  "Deviation from mean",
                }[q],
                key="map_quantity",
            )
            colorscale, map_opacity = render_map_display_controls("map")

    da_map = final_da(all_datasets.get(map_name))
    if da_map is None:
        with right:
            st.warning("Processing result not available for this file. Visit the Preprocessing page first.")
        return

    with right:
        _render_map_section(
            da_map, ds, map_name, loaded[map_name]["hash"], pipeline_params,
            quantity, colorscale, x_unit, laser_nm, map_opacity,
        )

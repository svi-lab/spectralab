# -*- coding: utf-8 -*-
"""Map Analysis page: spectral map parameters and heatmap visualization."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend._shared.dataset import SpectralDataset
from ..map_chart import make_map_fig

_UNIT_DISPLAY = {
    "RamanShift":   "cm⁻¹",
    "Wavenumber":   "cm⁻¹",
    "Nanometer":    "nm",
    "ElectronVolt": "eV",
}


def _default_pipeline_params() -> dict[str, Any]:
    """All-disabled pipeline params for cold map page loads (no preprocessing)."""
    return {
        "norm1_enabled": False, "norm1": {},
        "cd_enabled":    False, "cd":    {},
        "crr_enabled":   False, "crr":   {},
        "norm2_enabled": False, "norm2": {},
        "denoise_enabled": False, "denoise": {},
        "norm3_enabled": False, "norm3": {},
    }


def render_map_page() -> None:
    """Map Analysis page: map parameters (left) + plotly heatmap (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    all_finals: dict[str, Any] | None = st.session_state.get("sl_finals")

    # If finals aren't cached yet (user navigated here before Preprocessing page ran),
    # re-run with stored params or all-disabled defaults. The cache makes this free
    # once the Preprocessing page has been visited.
    if all_finals is None:
        from backend.pipeline import preprocess
        params = st.session_state.get("sl_pipeline_params", _default_pipeline_params())
        all_finals = {}
        with st.spinner("Preparing data…"):
            for name, entry in loaded.items():
                try:
                    _, da_final = preprocess(entry["dataset"], params)
                    all_finals[name] = da_final
                except Exception:
                    pass

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
        spectral_label = ds.spectral_dim.replace("_", " ")

        st.markdown('<p class="section-header">Spectral Range</p>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        lmin = mc1.number_input(
            f"λ min ({spectral_label})",
            value=float(round(ds.spec_min + (ds.spec_max - ds.spec_min) * 0.2, 1)),
            min_value=float(ds.spec_min), max_value=float(ds.spec_max),
            step=1.0, key="map_lmin",
        )
        lmax = mc2.number_input(
            f"λ max ({spectral_label})",
            value=float(round(ds.spec_min + (ds.spec_max - ds.spec_min) * 0.8, 1)),
            min_value=float(ds.spec_min), max_value=float(ds.spec_max),
            step=1.0, key="map_lmax",
        )

        if lmin >= lmax:
            st.warning("λ min must be less than λ max.")

        st.markdown('<p class="section-header">Display</p>', unsafe_allow_html=True)
        quantity = st.selectbox(
            "Quantity",
            ["integrated", "deviation"],
            format_func=lambda q: {
                "integrated": "Integrated intensity",
                "deviation":  "Deviation from mean",
            }[q],
            key="map_quantity",
        )
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Inferno", "Hot", "RdBu_r", "Turbo"],
            key="map_colorscale",
        )
        flip_y = st.checkbox(
            "Flip Y axis", key="map_flip_y",
            help="Toggle if the image appears upside-down relative to the heatmap.",
        )

    with right:
        if lmin >= lmax:
            return

        da_map = all_finals.get(map_name)
        if da_map is None:
            st.warning("Processing result not available for this file. Visit the Preprocessing page first.")
            return

        spectral_unit_display = (
            _UNIT_DISPLAY.get(ds.spectral_units) or ds.spectral_unit or "nm"
        )
        with st.spinner("Building map…"):
            fig_map = make_map_fig(
                da=da_map,
                image_arr=ds.image_arr,
                image_meta=ds.image_meta,
                lambda_min=lmin,
                lambda_max=lmax,
                quantity=quantity,
                colorscale=colorscale,
                title=map_name,
                flip_y=flip_y,
                spectral_unit=spectral_unit_display,
            )
        st.plotly_chart(fig_map, width="stretch")

        if ds.image_arr is None:
            st.caption("ℹ No white-light image (WHTL block) found in this file — heatmap only.")
        elif ds.image_meta is None:
            st.caption("ℹ Image found but EXIF geo-registration data missing — heatmap only.")

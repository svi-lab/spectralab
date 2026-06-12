# -*- coding: utf-8 -*-
"""Right panel: pipeline execution and tab rendering."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from backend.pipeline import preprocess
from backend._shared.dataset import SpectralDataset
from backend._shared.scan_geometry import get_scan_geometry
from backend._shared.scan_overlay import draw_scan_overlay
from .charts import convert_x, make_comparison_echarts, make_final_echarts, make_progress_echarts
from .controls import render_axis_controls, X_UNIT_OPTIONS, X_UNIT_FMT, UNIT_DEFAULT
from .map_chart import make_map_fig

_UNIT_DISPLAY = {
    "RamanShift":   "cm⁻¹",
    "Wavenumber":   "cm⁻¹",
    "Nanometer":    "nm",
    "ElectronVolt": "eV",
}


@st.cache_data
def _preprocess_cached(file_hash: str, _dataset: SpectralDataset, pipeline_params):
    return preprocess(_dataset, pipeline_params)


def _show_images(
    names: list[str],
    loaded: dict[str, Any],
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
                    arr = draw_scan_overlay(arr, ds.image_meta, geo,
                                           removed_mask=removed_mask)
                col.image(arr, use_container_width=True)


def _get_removed_mask(geo, ds, all_finals, name) -> np.ndarray | None:
    """Return bool mask (len = len(geo.xs)) marking CleanData-removed points."""
    if geo.shape != "points" or all_finals is None:
        return None
    da_final = all_finals.get(name)
    if da_final is None:
        return None
    if da_final.ndim == 3:
        # NaN-filled pixels in 3D map → rows align with meshgrid ravel order
        nan_mask = np.isnan(da_final.values).all(axis=-1)
        if not nan_mask.any():
            return None
        return nan_mask.ravel()
    if da_final.ndim == 2 and "point" in da_final.coords and "point" in ds.da.coords:
        # CleanData drops rows for 2D; missing point-index values = removed
        final_ids = set(da_final.coords["point"].values.tolist())
        mask = np.array([
            pid not in final_ids
            for pid in ds.da.coords["point"].values.tolist()
        ])
        return mask if mask.any() else None
    return None


def render_tabs(state: dict[str, Any]) -> None:
    """Run the preprocessing pipeline and render the Progress / Final / Map tabs."""
    loaded          = state["loaded"]
    pipeline_params = state["pipeline_params"]

    tab_progress, tab_final, tab_map = st.tabs(["Preprocessing", "Final", "Map"])

    # ── Run preprocessing ─────────────────────────────────────────────────────
    all_stages: dict[str, dict] = {}
    all_finals: dict[str, Any]  = {}
    errors: list[str]           = []

    with st.spinner(f"Processing {len(loaded)} file(s)…"):
        for name, entry in loaded.items():
            try:
                stages, da_final = _preprocess_cached(
                    entry["hash"], entry["dataset"], pipeline_params
                )
                all_stages[name] = stages
                all_finals[name] = da_final
            except Exception as exc:
                errors.append(f"{name}: {exc}")

    for err in errors:
        st.error(f"Processing error — {err}")
    if not all_finals:
        st.stop()

    multi = len(all_finals) > 1

    # Reference spectral axis — taken from first loaded file
    _ref_ds: SpectralDataset = next(iter(loaded.values()))["dataset"]

    _render_progress_tab(tab_progress, all_stages, all_finals, loaded, multi, _ref_ds)
    _render_final_tab(tab_final, all_finals, loaded, multi, _ref_ds)
    _render_map_tab(tab_map, loaded, all_finals)


# ---------------------------------------------------------------------------
# Private tab renderers
# ---------------------------------------------------------------------------

def _render_progress_tab(tab, all_stages, all_finals, loaded, multi, ref_ds: SpectralDataset):
    with tab:
        # Read current unit from session state so range defaults update when unit changes
        default_unit = UNIT_DEFAULT.get(ref_ds.spectral_units, "wavelength")
        current_unit = st.session_state.get("prog_x_unit", default_unit)

        # Compute display-unit bounds from reference dataset for From/To defaults
        x_native = ref_ds.da.coords[ref_ds.spectral_dim].values
        x_disp = convert_x(
            x_native, ref_ds.spectral_dim, current_unit,
            ref_ds.laser_nm, src_unit=ref_ds.spectral_unit,
            native_type=ref_ds.spectral_units,
        )
        disp_min = float(x_disp.min())
        disp_max = float(x_disp.max())

        # Single row: Spectral units | From | To | Laser (conditional) | Title
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
        _show_images(names_img, loaded, all_finals)


def _render_final_tab(tab, all_finals, loaded, multi, ref_ds: SpectralDataset):
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
            _show_images(list(loaded.keys()), loaded, all_finals)

        else:
            if multi:
                selected = st.selectbox(
                    "Select file", list(all_finals.keys()), key="final_file_select"
                )
            else:
                selected = next(iter(all_finals))

            # Use the selected file's own dataset for per-file axis accuracy
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
                make_final_echarts(
                    da_sel, title=selected,
                    color_by=color_by, n_bins=n_bins_val, step=int(step),
                    x_unit=x_unit, laser_nm=laser,
                    src_unit=sel_ds.spectral_unit, native_type=sel_ds.spectral_units,
                ),
                height="72vh", key="final_single",
            )
            _show_images([selected], loaded, all_finals)


def _render_map_tab(tab, loaded, all_finals):
    with tab:
        map_candidates = {
            name: entry for name, entry in loaded.items()
            if entry["dataset"].is_map
        }

        if not map_candidates:
            st.info(
                "No raster map files loaded. "
                "The Map tab works with 3D (row, column, spectral) DataArrays."
            )
            return

        if len(map_candidates) > 1:
            map_name = st.selectbox(
                "Select file", list(map_candidates.keys()), key="map_file_select"
            )
        else:
            map_name = next(iter(map_candidates))

        ds: SpectralDataset = map_candidates[map_name]["dataset"]
        da_map = all_finals.get(map_name)

        if da_map is None:
            st.warning("Processing failed for this file.")
            return

        spectral_label = ds.spectral_dim.replace("_", " ")

        mc1, mc2, mc3, mc4 = st.columns([1.2, 1.2, 1.5, 1])
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
        quantity = mc3.selectbox(
            "Quantity",
            ["integrated", "deviation"],
            format_func=lambda q: {
                "integrated": "Integrated intensity",
                "deviation":  "Deviation from mean",
            }[q],
            key="map_quantity",
        )
        colorscale = mc4.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Inferno", "Hot", "RdBu_r", "Turbo"],
            key="map_colorscale",
        )

        _adv_col, flip_col = st.columns([3, 1])
        flip_y = flip_col.checkbox(
            "Flip Y axis", key="map_flip_y",
            help="Toggle if the image appears upside-down relative to the heatmap.",
        )

        if lmin >= lmax:
            st.warning("λ min must be less than λ max.")
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
        st.plotly_chart(fig_map, width='stretch')

        if ds.image_arr is None:
            st.caption("ℹ No white-light image (WHTL block) found in this file — heatmap only.")
        elif ds.image_meta is None:
            st.caption("ℹ Image found but EXIF geo-registration data missing — heatmap only.")

# -*- coding: utf-8 -*-
"""Gaussian deconvolution page: manual multi-peak fitting with result statistics."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
from backend.peak_fitter import BandSpec, FitResult, PeakFitter, fit_map_gaussian

from ..charts import make_deconv_fit_echarts
from ..controls import render_axis_controls
from ..map_chart import make_scalar_map_fig
from ..pipeline_cache import default_pipeline_params, get_finals

_BAND_COLUMNS = ["label", "center_guess", "center_min", "center_max", "sigma_guess", "sigma_min", "sigma_max"]


def _default_bands_table(x: np.ndarray) -> list[dict]:
    center = float(np.median(x)) if len(x) else 0.0
    return [{
        "label": "Band 1", "center_guess": round(center, 2),
        "center_min": None, "center_max": None,
        "sigma_guess": None, "sigma_min": None, "sigma_max": None,
    }]


def _none_or_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _bands_from_table(rows: list[dict]) -> list[BandSpec]:
    bands: list[BandSpec] = []
    for i, row in enumerate(rows):
        center = _none_or_float(row.get("center_guess"))
        if center is None:
            continue
        bands.append(BandSpec(
            center_guess=center,
            center_min=_none_or_float(row.get("center_min")),
            center_max=_none_or_float(row.get("center_max")),
            sigma_guess=_none_or_float(row.get("sigma_guess")),
            sigma_min=_none_or_float(row.get("sigma_min")),
            sigma_max=_none_or_float(row.get("sigma_max")),
            label=(row.get("label") or None) or f"Band {i + 1}",
        ))
    return bands


def _fit_stats_rows(fit_result: FitResult) -> list[dict]:
    return [
        {
            "Band": b.label,
            "Center": round(b.center, 4),
            "Center stderr": None if b.center_stderr is None else round(b.center_stderr, 5),
            "Amplitude (area)": round(b.amplitude, 4),
            "Sigma": round(b.sigma, 5),
            "FWHM": round(b.fwhm, 5),
            "% Area": round(b.area_pct, 2),
        }
        for b in fit_result.bands
    ]


def _fit_stats_csv(fit_result: FitResult) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "label", "center", "center_stderr", "amplitude", "amplitude_stderr",
        "sigma", "sigma_stderr", "fwhm", "fwhm_stderr", "area", "area_pct",
    ])
    for b in fit_result.bands:
        writer.writerow([
            b.label, b.center, b.center_stderr, b.amplitude, b.amplitude_stderr,
            b.sigma, b.sigma_stderr, b.fwhm, b.fwhm_stderr, b.area, b.area_pct,
        ])
    writer.writerow([])
    writer.writerow(["r_squared", fit_result.r_squared])
    writer.writerow(["reduced_chi_square", fit_result.reduced_chi_square])
    writer.writerow(["aic", fit_result.aic])
    writer.writerow(["bic", fit_result.bic])
    return buf.getvalue().encode("utf-8")


def _batch_result_csv(batch_result, labels: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "row", "column", "band", "center", "amplitude", "sigma", "fwhm", "area",
        "r_squared", "reduced_chi_square", "success",
    ])
    n_row, n_col = batch_result.r_squared_map.shape
    for r in range(n_row):
        for c in range(n_col):
            for label in labels:
                band = batch_result.band_results[label]
                writer.writerow([
                    r, c, label,
                    band["center"][r, c], band["amplitude"][r, c], band["sigma"][r, c],
                    band["fwhm"][r, c], band["area"][r, c],
                    batch_result.r_squared_map[r, c], batch_result.reduced_chi_square_map[r, c],
                    bool(batch_result.success_map[r, c]),
                ])
    return buf.getvalue().encode("utf-8")


def render_deconvolution_page() -> None:
    """Deconvolution page: target + band parameters (left), fit results (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    pipeline_params = st.session_state.get("sl_pipeline_params") or default_pipeline_params()
    with st.spinner("Preparing data…"):
        _, all_finals, _errors = get_finals(loaded, pipeline_params)

    left, right = st.columns([1, 2], gap="medium")

    with left:
        if len(loaded) > 1:
            file_name = st.selectbox("Select file", list(loaded.keys()), key="deconv_file_select")
        else:
            file_name = next(iter(loaded))
        ds: SpectralDataset = loaded[file_name]["dataset"]

    if ds.measurement_kind != "PL":
        with left:
            st.info("Deconvolution is available for PL data only (Nanometer/ElectronVolt axes).")
        with right:
            st.info("Deconvolution is available for PL data only (Nanometer/ElectronVolt axes).")
        return

    da_final = all_finals.get(file_name)
    if da_final is None:
        with right:
            st.warning("Processing result not available for this file. Visit the Preprocessing page first.")
        return

    spectral_dim = da_final.dims[-1]
    spatial_dims = [d for d in da_final.dims if d != spectral_dim]

    with left:
        st.markdown('<p class="section-header">Display</p>', unsafe_allow_html=True)
        x_unit, laser_nm = render_axis_controls(
            "deconv", ds.laser_nm, native_type=ds.spectral_units,
        )

        st.markdown('<p class="section-header">Target Spectrum</p>', unsafe_allow_html=True)

        target_options = ["Mean spectrum"]
        if da_final.ndim == 3:
            target_options.append("Single pixel")
        nmf_result = st.session_state.get("sl_nmf_result")
        nmf_available = bool(nmf_result and nmf_result["file_name"] == file_name)
        if nmf_available:
            target_options.append("NMF component")

        target_mode = st.radio("Fit target", target_options, key="deconv_target_mode")

        target_x: np.ndarray
        target_y: np.ndarray

        if target_mode == "Single pixel":
            n_row = da_final.sizes[spatial_dims[0]]
            n_col = da_final.sizes[spatial_dims[1]]
            c1, c2 = st.columns(2)
            row_idx = c1.number_input("Row index", 0, n_row - 1, 0, key="deconv_row_idx")
            col_idx = c2.number_input("Column index", 0, n_col - 1, 0, key="deconv_col_idx")
            target_da = da_final.isel({
                spatial_dims[0]: int(row_idx), spatial_dims[1]: int(col_idx),
            })
            target_x = target_da.coords[spectral_dim].values
            target_y = target_da.values
            if bool(np.all(np.isnan(target_y))):
                st.warning("This pixel is NaN (dead/oversaturated). Pick another.")
        elif target_mode == "NMF component":
            n_comp = nmf_result["components"].shape[0]
            comp_idx = st.selectbox(
                "Component", range(n_comp),
                format_func=lambda i: f"Component {i + 1}",
                key="deconv_nmf_comp_select",
            )
            target_x = nmf_result["spectral_coords"]
            target_y = nmf_result["components"][comp_idx]
        else:  # Mean spectrum
            target_da = da_final.mean(spatial_dims, skipna=True) if spatial_dims else da_final
            target_x = target_da.coords[spectral_dim].values
            target_y = target_da.values

        unit_label = "eV" if ds.spectral_units == "ElectronVolt" else "nm"
        st.markdown('<p class="section-header">Band Parameters</p>', unsafe_allow_html=True)
        st.caption(f"Positions are in native units ({unit_label}), matching this file's stored axis.")
        bands_table = st.data_editor(
            st.session_state.get("deconv_bands_table", _default_bands_table(target_x)),
            num_rows="dynamic",
            column_config={
                "label":        st.column_config.TextColumn("Label"),
                "center_guess": st.column_config.NumberColumn(f"Center guess ({unit_label})", required=True),
                "center_min":   st.column_config.NumberColumn(f"Center min ({unit_label})"),
                "center_max":   st.column_config.NumberColumn(f"Center max ({unit_label})"),
                "sigma_guess":  st.column_config.NumberColumn("Sigma guess (auto if blank)"),
                "sigma_min":    st.column_config.NumberColumn("Sigma min"),
                "sigma_max":    st.column_config.NumberColumn("Sigma max"),
            },
            column_order=_BAND_COLUMNS,
            key="deconv_bands_editor",
        )
        st.session_state["deconv_bands_table"] = bands_table

        fit_clicked = st.button("Fit", key="deconv_fit_button", type="primary")

        st.divider()
        st.markdown('<p class="section-header">Full-Map Batch Fit</p>', unsafe_allow_html=True)
        if da_final.ndim == 3:
            st.caption(
                "Fits every pixel independently, warm-starting each pixel from its "
                "neighbor's converged parameters. May take from seconds to minutes "
                "depending on map size and band count."
            )
            batch_clicked = st.button("Fit entire map", key="deconv_batch_fit_button")
        else:
            batch_clicked = False
            st.caption("Full-map batch fit requires a map-scan file.")

    with right:
        if fit_clicked:
            bands = _bands_from_table(bands_table)
            if not bands:
                st.error("Add at least one band with a center guess before fitting.")
            else:
                try:
                    fit_result = PeakFitter().fit(target_x, target_y, bands)
                except (ValueError, NotImplementedError) as exc:
                    st.error(f"Fit failed: {exc}")
                else:
                    st.session_state["sl_deconv_result"] = fit_result
                    st.session_state["sl_deconv_result_file"] = file_name

        fit_result = st.session_state.get("sl_deconv_result")
        if fit_result is not None and st.session_state.get("sl_deconv_result_file") == file_name:
            st_echarts(
                make_deconv_fit_echarts(
                    fit_result, spectral_dim,
                    x_unit=x_unit, laser_nm=laser_nm,
                    src_unit=ds.spectral_unit, native_type=ds.spectral_units,
                ),
                height="450px",
            )
            if not fit_result.success:
                st.warning(f"Solver did not report success: {fit_result.message}")
            st.caption(
                f"R² = {fit_result.r_squared:.4f}  ·  "
                f"reduced χ² = {fit_result.reduced_chi_square:.4g}  ·  "
                f"AIC = {fit_result.aic:.1f}  ·  BIC = {fit_result.bic:.1f}"
            )
            st.dataframe(_fit_stats_rows(fit_result), width="stretch")
            st.download_button(
                "Download fit statistics (CSV)",
                _fit_stats_csv(fit_result),
                file_name=f"{file_name}_deconv_fit.csv",
                mime="text/csv",
                key="deconv_download_single",
            )

        if batch_clicked:
            bands = _bands_from_table(bands_table)
            if not bands:
                st.error("Add at least one band with a center guess before fitting.")
            else:
                progress_bar = st.progress(0.0)

                def _cb(done: int, total: int) -> None:
                    progress_bar.progress(done / total)

                with st.spinner("Fitting every pixel…"):
                    batch_result = fit_map_gaussian(da_final, bands, progress_callback=_cb)
                progress_bar.empty()
                st.session_state["sl_deconv_batch_result"] = batch_result
                st.session_state["sl_deconv_batch_labels"] = [b.label or f"Band {i+1}" for i, b in enumerate(bands)]
                st.session_state["sl_deconv_batch_file"] = file_name

        batch_result = st.session_state.get("sl_deconv_batch_result")
        if batch_result is not None and st.session_state.get("sl_deconv_batch_file") == file_name:
            st.markdown("**Full-map fit results**")
            st.caption(
                f"Fitted {batch_result.n_fitted} pixels · "
                f"skipped {batch_result.n_skipped_nan} NaN pixels · "
                f"{batch_result.n_failed} failed fits"
            )
            labels = st.session_state.get("sl_deconv_batch_labels", list(batch_result.band_results.keys()))
            c1, c2 = st.columns(2)
            band_label = c1.selectbox("Band", labels, key="deconv_batch_band_select")
            param_name = c2.selectbox(
                "Parameter", ["center", "amplitude", "sigma", "fwhm", "area"],
                key="deconv_batch_param_select",
            )
            z = batch_result.band_results[band_label][param_name]
            row_coords = da_final.coords[spatial_dims[0]].values
            col_coords = da_final.coords[spatial_dims[1]].values
            fig = make_scalar_map_fig(
                z, row_coords, col_coords, ds.image_arr, ds.image_meta,
                cbar_label=f"{band_label} {param_name}",
                title=f"{file_name} — {band_label} {param_name}",
            )
            st.plotly_chart(fig, width="stretch", height=550)
            st.download_button(
                "Download per-pixel parameters (CSV)",
                _batch_result_csv(batch_result, labels),
                file_name=f"{file_name}_batch_fit.csv",
                mime="text/csv",
                key="deconv_download_batch",
            )

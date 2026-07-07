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
from backend.peak_fitter import (
    BandPreset,
    BandSpec,
    FitResult,
    PeakFitter,
    fit_map_gaussian,
    get_preset_bands,
    list_preset_materials,
)
from ..export_utils import batch_fit_to_npz, fit_curves_to_npz

from ..charts import convert_x_to_native, make_deconv_fit_echarts, make_deconv_preview_echarts
from ..controls import render_axis_controls
from ..map_chart import make_scalar_map_fig
from ..pipeline_cache import default_pipeline_params, get_finals

_BAND_COLUMNS = ["label", "center_guess", "center_min", "center_max", "sigma_guess", "sigma_min", "sigma_max"]

# Every preset band gets a +/- 20 nm center bound by default, on top of its literature
# position, so an initial fit doesn't let a peak wander into a neighboring one.
_PRESET_BOUND_HALF_WIDTH_NM = 20.0

# Bound to zrender (not the chart-level "click") so it fires on blank canvas too, not just
# on rendered lines/points. Returns undefined for clicks outside the grid (toolbox, legend,
# title) so those don't round-trip to Python at all.
_CLICK_TO_ADD_BAND_JS = """
function (params) {
    var pixel = [params.offsetX, params.offsetY];
    if (!chart.containPixel('grid', pixel)) { return; }
    var dataPoint = chart.convertFromPixel('grid', pixel);
    return {x: dataPoint[0]};
}
"""


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


def _reset_bands_table(rows: list[dict]) -> None:
    """Replace the band table and drop the editor's cached widget state.

    st.data_editor caches its rows under its own `key` once rendered, so just
    changing the `value` argument on a later rerun does not reliably refresh
    what's on screen — the cached widget state has to be cleared too.
    """
    st.session_state["deconv_bands_table"] = rows
    st.session_state.pop("deconv_bands_editor", None)


def _append_bands(new_rows: list[dict]) -> None:
    current = st.session_state.get("deconv_bands_table") or []
    _reset_bands_table([*current, *new_rows])


def _preset_band_half_widths_nm(presets: tuple[BandPreset, ...]) -> dict[float, float]:
    """Cap each band's default +/- half-width so neighboring preset bands' windows
    never overlap.

    With a flat +/-20 nm bound, literature positions closer than 40 nm apart (common
    in these tables — e.g. ZnO:Al's 437/440 nm pair) get overlapping allowed ranges,
    which lets the optimizer swap which Gaussian claims which literature position
    (two bands' fitted centers end up nearer each other's neighbor than their own
    label). Halving the gap to the nearest neighbor on each side makes that swap
    mathematically impossible: no two bands' windows can ever touch a third band's.
    """
    sorted_nm = sorted(p.wavelength_nm for p in presets)
    half_widths: dict[float, float] = {}
    for i, nm in enumerate(sorted_nm):
        candidates = [_PRESET_BOUND_HALF_WIDTH_NM]
        if i > 0:
            candidates.append((nm - sorted_nm[i - 1]) / 2)
        if i < len(sorted_nm) - 1:
            candidates.append((sorted_nm[i + 1] - nm) / 2)
        half_widths[nm] = min(candidates)
    return half_widths


def _preset_rows_to_table_rows(presets: tuple[BandPreset, ...], native_type: str) -> list[dict]:
    half_widths = _preset_band_half_widths_nm(presets)
    rows: list[dict] = []
    for p in presets:
        half = half_widths[p.wavelength_nm]
        nm_lo = p.wavelength_nm - half
        nm_hi = p.wavelength_nm + half
        if native_type == "ElectronVolt":
            center = p.energy_ev
            # nm -> eV is inversely proportional, so the longer-wavelength edge maps to
            # the lower-energy bound. 1239.84 is the same hc[eV*nm] constant charts.convert_x uses.
            lo, hi = 1239.84 / nm_hi, 1239.84 / nm_lo
        else:  # "Nanometer" — the only other native type this PL-only page allows
            center = p.wavelength_nm
            lo, hi = nm_lo, nm_hi
        rows.append({
            "label": p.label, "center_guess": center,
            "center_min": round(lo, 4), "center_max": round(hi, 4),
            "sigma_guess": None, "sigma_min": None, "sigma_max": None,
        })
    return rows


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
        with st.container(border=True):
            st.markdown('<p class="section-header">Display</p>', unsafe_allow_html=True)
            x_unit, laser_nm = render_axis_controls(
                "deconv", ds.laser_nm, native_type=ds.spectral_units,
            )

        with st.container(border=True):
            st.markdown('<p class="section-header">Target Spectrum</p>', unsafe_allow_html=True)

            target_options = ["Mean spectrum"]
            if da_final.ndim == 3:
                target_options.append("Single pixel")
            nmf_result = st.session_state.get("sl_nmf_result")
            nmf_available = bool(nmf_result and nmf_result["file_name"] == file_name)
            if nmf_available:
                target_options.append("NMF component")
            mcr_result = st.session_state.get("sl_mcr_result")
            mcr_available = bool(mcr_result and mcr_result["file_name"] == file_name)
            if mcr_available:
                target_options.append("MCR component")

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
            elif target_mode == "MCR component":
                n_comp = mcr_result["components"].shape[0]
                comp_idx = st.selectbox(
                    "Component", range(n_comp),
                    format_func=lambda i: f"Component {i + 1}",
                    key="deconv_mcr_comp_select",
                )
                target_x = mcr_result["spectral_coords"]
                target_y = mcr_result["components"][comp_idx]
            else:  # Mean spectrum
                target_da = da_final.mean(spatial_dims, skipna=True) if spatial_dims else da_final
                target_x = target_da.coords[spectral_dim].values
                target_y = target_da.values

        unit_label = "eV" if ds.spectral_units == "ElectronVolt" else "nm"
        with st.container(border=True):
            st.markdown('<p class="section-header">Band Parameters</p>', unsafe_allow_html=True)
            st.caption(f"Positions are in native units ({unit_label}), matching this file's stored axis.")

            preset_choice = st.selectbox(
                "Load preset", ["— none —", *list_preset_materials()], key="deconv_preset_select",
            )
            if preset_choice != "— none —":
                presets = get_preset_bands(preset_choice)
                st.caption(
                    f"Each band loads with a center bound of up to ±{_PRESET_BOUND_HALF_WIDTH_NM:.0f} nm "
                    "around its literature position, narrowed near closely-spaced neighbors so "
                    "bands can't swap which peak they claim."
                )
                st.dataframe(
                    [
                        {
                            "Label": p.label,
                            "λ (nm)": p.wavelength_nm,
                            "E (eV)": p.energy_ev,
                            "Assignment": p.assignment + (" (tentative)" if p.tentative else ""),
                        }
                        for p in presets
                    ],
                    width="stretch",
                    hide_index=True,
                )
                if st.button(f"Add {len(presets)} bands from {preset_choice}", key="deconv_add_preset"):
                    _append_bands(_preset_rows_to_table_rows(presets, ds.spectral_units))

            qc1, qc2, qc3, qc4 = st.columns([2, 2, 1, 1])
            qa_center = qc1.number_input(
                f"Quick-add center ({unit_label})",
                value=float(np.median(target_x)) if len(target_x) else 0.0,
                key="deconv_quick_add_center",
            )
            qa_label = qc2.text_input("Label (optional)", key="deconv_quick_add_label")
            qc3.markdown("<br>", unsafe_allow_html=True)
            if qc3.button("Add band", key="deconv_quick_add_button"):
                _append_bands([{
                    "label": qa_label or None, "center_guess": qa_center,
                    "center_min": None, "center_max": None,
                    "sigma_guess": None, "sigma_min": None, "sigma_max": None,
                }])
            qc4.markdown("<br>", unsafe_allow_html=True)
            if qc4.button("Clear all", key="deconv_clear_bands_button"):
                _reset_bands_table([])

            current_rows = st.session_state.get("deconv_bands_table") or []
            if current_rows:
                def _row_label(i: int) -> str:
                    row = current_rows[i]
                    label = row.get("label") or f"Band {i + 1}"
                    center = _none_or_float(row.get("center_guess"))
                    return f"{label} ({center:.2f} {unit_label})" if center is not None else label

                rc1, rc2 = st.columns([3, 1])
                to_remove = rc1.multiselect(
                    "Remove bands", options=list(range(len(current_rows))),
                    format_func=_row_label, key="deconv_remove_band_select",
                )
                rc2.markdown("<br>", unsafe_allow_html=True)
                if rc2.button("Remove", key="deconv_remove_band_button", disabled=not to_remove):
                    remaining = [row for i, row in enumerate(current_rows) if i not in to_remove]
                    _reset_bands_table(remaining)
                    st.session_state.pop("deconv_remove_band_select", None)
                    if st.session_state.get("sl_deconv_result_file") == file_name:
                        remaining_bands = _bands_from_table(remaining)
                        if remaining_bands:
                            try:
                                new_fit_result = PeakFitter().fit(target_x, target_y, remaining_bands)
                            except (ValueError, NotImplementedError) as exc:
                                st.error(f"Fit failed: {exc}")
                            else:
                                st.session_state["sl_deconv_result"] = new_fit_result
                        else:
                            st.session_state.pop("sl_deconv_result", None)
                            st.session_state.pop("sl_deconv_result_file", None)
                    st.rerun()

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

        with st.container(border=True):
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
        has_fit = fit_result is not None and st.session_state.get("sl_deconv_result_file") == file_name

        fit_title = st.text_input("Chart title", value="Peak Deconvolution", key="deconv_fit_title")
        st.caption("Click anywhere on the plot to drop a new band there and (re)fit.")
        if has_fit:
            chart_options = make_deconv_fit_echarts(
                fit_result, spectral_dim,
                title=fit_title,
                x_unit=x_unit, laser_nm=laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )
        else:
            band_centers_native = [
                c for row in bands_table
                if (c := _none_or_float(row.get("center_guess"))) is not None
            ]
            chart_options = make_deconv_preview_echarts(
                target_x, target_y, spectral_dim, band_centers_native,
                title=fit_title,
                x_unit=x_unit, laser_nm=laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )
        chart_value = st_echarts(
            chart_options,
            height="72vh",
            events={"zr:click": _CLICK_TO_ADD_BAND_JS},
            key="deconv_fit_chart",
        )
        # "chart_event" is a Streamlit v2 *trigger* value: it holds our JS handler's
        # raw return value and auto-resets to None after this script run, so a plain
        # not-None check is exactly-once per click -- no manual dedup needed.
        click_value = (chart_value or {}).get("chart_event")
        if click_value is not None:
            x_native = convert_x_to_native(
                click_value["x"], spectral_dim, x_unit, laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )
            _append_bands([{
                "label": None, "center_guess": round(x_native, 4),
                "center_min": None, "center_max": None,
                "sigma_guess": None, "sigma_min": None, "sigma_max": None,
            }])
            new_bands = _bands_from_table(st.session_state["deconv_bands_table"])
            if new_bands:
                try:
                    new_fit_result = PeakFitter().fit(target_x, target_y, new_bands)
                except (ValueError, NotImplementedError) as exc:
                    st.error(f"Fit failed: {exc}")
                else:
                    st.session_state["sl_deconv_result"] = new_fit_result
                    st.session_state["sl_deconv_result_file"] = file_name
            st.rerun()

        if has_fit:
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
            st.download_button(
                "Export fit curves (.npz)",
                fit_curves_to_npz(fit_result),
                file_name=f"{file_name}_deconv_curves.npz",
                key="deconv_export_curves_npz",
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
            st.download_button(
                "Export parameter maps (.npz)",
                batch_fit_to_npz(
                    batch_result, labels,
                    da_final.coords[spatial_dims[0]].values,
                    da_final.coords[spatial_dims[1]].values,
                ),
                file_name=f"{file_name}_batch_fit.npz",
                key="deconv_export_batch_npz",
            )

# -*- coding: utf-8 -*-
"""Gaussian deconvolution page: manual multi-peak fitting with result statistics."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
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

from ..charts import convert_x, convert_x_to_native, make_deconv_fit_echarts, make_deconv_preview_echarts
from ..controls import render_axis_controls, render_map_display_controls
from ..map_chart import make_scalar_map_fig
from ..pipeline_cache import default_pipeline_params, final_da, get_finals

_BAND_COLUMNS = ["label", "center_guess", "center_min", "center_max", "sigma_guess", "sigma_min", "sigma_max"]
_NUMERIC_BAND_COLUMNS = _BAND_COLUMNS[1:]

# The band editor is remounted under a fresh key whenever the display unit changes or
# the table is edited programmatically (preset load / chart click / clear), so the grid
# always shows the current rows -- st.data_editor caches rows under its own key and
# ignores a changed `value` argument otherwise.
_EDITOR_KEY_PREFIX = "deconv_bands_editor_"

# Editing precision per display unit. `step` is what actually fixes the "can only type
# whole numbers" problem: st.column_config.NumberColumn falls back to integer stepping
# unless the column resolves to a float dtype *and* a sub-unit step is given.
_UNIT_SPEC: dict[str, dict[str, Any]] = {
    "energy":      {"short": "eV",   "step": 0.0001, "format": "%.4f"},
    "wavelength":  {"short": "nm",   "step": 0.01,   "format": "%.2f"},
    "wavenumber":  {"short": "cm⁻¹", "step": 0.1,    "format": "%.1f"},
    "raman_shift": {"short": "cm⁻¹", "step": 0.1,    "format": "%.1f"},
}

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
    return None if not np.isfinite(f) else f


def _none_or_str(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v).strip() or None


# ---------------------------------------------------------------------------
# Band rows: canonical eV storage <-> the unit the Display selector shows
# ---------------------------------------------------------------------------

@dataclass
class _BandUnits:
    """Maps band parameters between the page's canonical eV storage and whichever
    unit the Display selector is currently showing.

    Band rows are always *stored* in eV, because fitting always runs on the energy
    axis — but the user types the number they read off the chart, so every value
    crossing the editor boundary passes through here. Two wrinkles this handles:

    * nm and Raman shift run *opposite* to eV, so a (min, max) pair has to swap
      slots on the way across — an eV lower bound is a nm upper bound.
    * a sigma is a width, not a position, so it maps through the local scale at its
      own band's center rather than pointwise.
    """

    spectral_dim: str
    x_unit: str
    laser_nm: float | None
    src_unit: str
    short: str = field(init=False)
    step: float = field(init=False)
    number_format: str = field(init=False)
    inverts: bool = field(init=False)

    def __post_init__(self) -> None:
        spec = _UNIT_SPEC.get(self.x_unit, _UNIT_SPEC["energy"])
        self.short = spec["short"]
        self.step = spec["step"]
        self.number_format = spec["format"]
        # Probed rather than hard-coded per unit, so it stays correct if convert_x
        # ever gains a unit.
        self.inverts = self.from_ev(2.0) < self.from_ev(1.0)

    def from_ev(self, ev: float) -> float:
        return float(convert_x(
            np.asarray([ev], dtype=float), self.spectral_dim, self.x_unit, self.laser_nm,
            src_unit=self.src_unit, native_type="ElectronVolt",
        )[0])

    def to_ev(self, disp: float) -> float:
        return float(convert_x_to_native(
            disp, self.spectral_dim, self.x_unit, self.laser_nm,
            src_unit=self.src_unit, native_type="ElectronVolt",
        ))

    def _scale_at(self, center_ev: float) -> float | None:
        """|d(display)/d(eV)| at ``center_ev``, by central difference.

        Widths convert through this local scale rather than by mapping the band's own
        ±σ edges: the edge mapping is exact for the edges but *not invertible* (the
        eV-symmetric interval [E−σ, E+σ] is not symmetric in nm about λ(E), so the
        round trip drifts — measured 1.3% per pass on a 0.2 eV σ). A single scale
        factor evaluated at the center is a linearization, but it inverts exactly,
        which matters far more here: these values round-trip through the editor on
        every rerun.
        """
        delta = max(abs(center_ev) * 1e-6, 1e-9)
        try:
            span = abs(self.from_ev(center_ev + delta) - self.from_ev(center_ev - delta))
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return None
        scale = span / (2 * delta)
        return scale if np.isfinite(scale) and scale > 0 else None

    def width_from_ev(self, sigma_ev: float | None, center_ev: float | None) -> float | None:
        if sigma_ev is None or center_ev is None or self.x_unit == "energy":
            return sigma_ev
        scale = self._scale_at(center_ev)
        return None if scale is None else sigma_ev * scale

    def width_to_ev(self, sigma_disp: float | None, center_disp: float | None) -> float | None:
        if sigma_disp is None or center_disp is None or self.x_unit == "energy":
            return sigma_disp
        try:
            scale = self._scale_at(self.to_ev(center_disp))
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return None
        return None if scale is None else sigma_disp / scale


def _row_to_display(units: _BandUnits, row: dict) -> dict:
    """One canonical (eV) band row -> the row shown in the editor."""
    center = _none_or_float(row.get("center_guess"))
    lo = _none_or_float(row.get("center_min"))
    hi = _none_or_float(row.get("center_max"))
    d_lo = units.from_ev(lo) if lo is not None else None
    d_hi = units.from_ev(hi) if hi is not None else None
    if units.inverts:
        d_lo, d_hi = d_hi, d_lo
    out: dict[str, Any] = {
        "label": _none_or_str(row.get("label")),
        "center_guess": units.from_ev(center) if center is not None else None,
        "center_min": d_lo,
        "center_max": d_hi,
    }
    for key in ("sigma_guess", "sigma_min", "sigma_max"):
        out[key] = units.width_from_ev(_none_or_float(row.get(key)), center)
    # Deliberately *not* rounded to the column's display precision: `format` already
    # controls what the cell shows, while the value itself keeps full precision, so
    # eV -> display -> eV is exact and merely switching the Display unit back and
    # forth leaves the stored centers bit-identical.
    return out


def _row_from_display(units: _BandUnits, row: dict) -> dict:
    """One editor row -> the canonical (eV) band row. Inverse of :func:`_row_to_display`."""
    center = _none_or_float(row.get("center_guess"))
    lo = _none_or_float(row.get("center_min"))
    hi = _none_or_float(row.get("center_max"))
    ev_lo = units.to_ev(lo) if lo is not None else None
    ev_hi = units.to_ev(hi) if hi is not None else None
    if units.inverts:
        ev_lo, ev_hi = ev_hi, ev_lo
    out: dict[str, Any] = {
        "label": _none_or_str(row.get("label")),
        "center_guess": units.to_ev(center) if center is not None else None,
        "center_min": ev_lo,
        "center_max": ev_hi,
    }
    for key in ("sigma_guess", "sigma_min", "sigma_max"):
        out[key] = units.width_to_ev(_none_or_float(row.get(key)), center)
    return out


def _display_frame(units: _BandUnits, rows: list[dict]) -> pd.DataFrame:
    """Build the editor's DataFrame with every column's dtype pinned.

    Pinning matters for more than tidiness. A list of dicts whose ``sigma_*`` values
    are all ``None`` gives pandas an *object* column, which Streamlit resolves to
    ``ColumnDataKind.EMPTY`` — a NumberColumn over an EMPTY column has no float hint
    and falls back to integer stepping, so decimals typed into it get truncated.
    Naming the columns explicitly also keeps an empty table (after "Clear all") a
    usable 7-column grid instead of a zero-column one with nothing to type into.
    """
    df = pd.DataFrame([_row_to_display(units, r) for r in rows], columns=_BAND_COLUMNS)
    df["label"] = df["label"].astype("string")
    for col in _NUMERIC_BAND_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _bands_digest(rows: list[dict]) -> str:
    """Cheap identity for the staged band table, to tell whether the displayed fit
    is still the fit for these bands."""
    return repr([tuple(_none_or_float(r.get(c)) for c in _NUMERIC_BAND_COLUMNS) for r in rows])


# ---------------------------------------------------------------------------
# Band table state (canonical eV; never display units)
# ---------------------------------------------------------------------------

def _set_bands(rows: list[dict]) -> None:
    """Replace the band table and force the editor to remount on the next run.

    Two keys, deliberately: ``deconv_bands_source`` is what the grid renders from and
    only ever changes here, while ``deconv_bands_table`` is the resolved result the
    rest of the page reads (grid delta already applied) — see :func:`_sync_editor_source`
    for why writing the resolved rows back into the source would duplicate rows.
    Bumping the revision is what actually makes a programmatic edit visible: the
    editor caches its rows under its own key and ignores a changed ``data`` argument
    unless that key changes.
    """
    st.session_state["deconv_bands_source"] = rows
    st.session_state["deconv_bands_table"] = rows
    st.session_state["deconv_bands_rev"] = st.session_state.get("deconv_bands_rev", 0) + 1


def _append_bands(new_rows: list[dict]) -> None:
    current = st.session_state.get("deconv_bands_table") or []
    _set_bands([*current, *new_rows])


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


def _preset_rows_to_table_rows(presets: tuple[BandPreset, ...]) -> list[dict]:
    """Canonical band rows, i.e. always eV — fitting on this page always runs in energy
    space, independent of the file's native storage unit and of whichever unit the
    editor happens to be showing (see :class:`_BandUnits`)."""
    half_widths = _preset_band_half_widths_nm(presets)
    rows: list[dict] = []
    for p in presets:
        half = half_widths[p.wavelength_nm]
        nm_lo = p.wavelength_nm - half
        nm_hi = p.wavelength_nm + half
        center = p.energy_ev
        # nm -> eV is inversely proportional, so the longer-wavelength edge maps to
        # the lower-energy bound. 1239.84 is the same hc[eV*nm] constant charts.convert_x uses.
        lo, hi = 1239.84 / nm_hi, 1239.84 / nm_lo
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
            label=_none_or_str(row.get("label")) or f"Band {i + 1}",
        ))
    return bands


def _sync_editor_source(units: _BandUnits) -> str:
    """Return the band editor's widget key, refreshing the rows it renders from
    whenever that key is about to change.

    st.data_editor keeps the user's adds/edits/deletes as a *delta* against its
    ``data`` argument and replays that delta on every rerun. So the resolved rows must
    never be written back into ``data`` under an unchanged key — the delta would apply
    a second time and every row added through the grid would duplicate itself. The
    source is therefore refreshed only at the two moments the key changes: a
    programmatic edit (which bumps the revision) or a display-unit switch (which has
    to remount the grid to relabel it in the new unit).
    """
    if st.session_state.get("deconv_bands_unit") != units.x_unit:
        st.session_state["deconv_bands_source"] = st.session_state.get("deconv_bands_table") or []
        st.session_state["deconv_bands_unit"] = units.x_unit
    return f"{_EDITOR_KEY_PREFIX}{units.x_unit}_{st.session_state.get('deconv_bands_rev', 0)}"


def _render_band_editor(units: _BandUnits, file_name: str) -> tuple[list[dict], bool]:
    """Full-width Band Parameters card. Returns (canonical eV rows, fit_clicked).

    The grid is the single editing surface: its own trailing blank row and row-delete
    replace what used to be a separate quick-add number input and a "Remove bands"
    multiselect. Those three widget groups all edited the same list keyed by list
    index, so any row that moved left the others pointing at the wrong band.
    """
    with st.container(border=True):
        st.markdown('<p class="section-header">Band Parameters</p>', unsafe_allow_html=True)

        head = st.columns([3, 2, 1, 2, 5], vertical_alignment="bottom")
        preset_choice = head[0].selectbox(
            "Load preset", ["— none —", *list_preset_materials()], key="deconv_preset_select",
        )
        presets = get_preset_bands(preset_choice) if preset_choice != "— none —" else ()
        head[1].button(
            f"Add {len(presets)} bands" if presets else "Add preset bands",
            key="deconv_add_preset", disabled=not presets, width="stretch",
            on_click=_append_bands, args=(_preset_rows_to_table_rows(presets),),
        )
        head[2].button(
            "Clear", key="deconv_clear_bands_button", width="stretch",
            on_click=_set_bands, args=([],),
        )
        fit_clicked = head[3].button(
            "Fit", key="deconv_fit_button", type="primary", width="stretch",
        )

        if presets:
            with st.expander(f"{preset_choice} literature positions", expanded=False):
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

        editor_key = _sync_editor_source(units)
        for stale in [
            k for k in st.session_state
            if isinstance(k, str) and k.startswith(_EDITOR_KEY_PREFIX) and k != editor_key
        ]:
            st.session_state.pop(stale, None)

        u = units.short

        def _num(label: str, help_text: str | None = None) -> Any:
            return st.column_config.NumberColumn(
                label, step=units.step, format=units.number_format, help=help_text,
            )

        edited = st.data_editor(
            _display_frame(units, st.session_state.get("deconv_bands_source") or []),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "label": st.column_config.TextColumn("Label", width="medium"),
                "center_guess": _num(f"Center ({u})", "Starting position. Rows left blank here are ignored."),
                "center_min": _num(f"Center min ({u})", "Blank = unconstrained."),
                "center_max": _num(f"Center max ({u})", "Blank = unconstrained."),
                "sigma_guess": _num(f"σ guess ({u})", "Blank = estimated from the data."),
                "sigma_min": _num(f"σ min ({u})", "Blank = unconstrained."),
                "sigma_max": _num(f"σ max ({u})", "Blank = unconstrained."),
            },
            column_order=_BAND_COLUMNS,
            key=editor_key,
        )
        rows = [_row_from_display(units, r) for r in edited.to_dict("records")]
        st.session_state["deconv_bands_table"] = rows

        st.caption(
            f"Positions are in {u}, following the **Display → Spectral units** selector below — "
            "switch it to type in a different unit. Bands are stored and fitted in energy space "
            "either way, so switching relabels the same physical bands. Add a band by typing into "
            "the trailing blank row or by clicking the chart; delete by selecting rows and pressing "
            "the trash icon."
        )
        if (
            st.session_state.get("sl_deconv_result") is not None
            and st.session_state.get("sl_deconv_result_file") == file_name
            and st.session_state.get("sl_deconv_result_digest") != _bands_digest(rows)
        ):
            st.caption("⚠️ Band table changed since the last fit — press **Fit** to update the chart.")

    return rows, fit_clicked


def _run_fit(target_x: np.ndarray, target_y: np.ndarray, rows: list[dict], file_name: str) -> None:
    """Fit the staged bands and store the result (plus the band digest the stale-fit
    notice compares against). Errors are surfaced, not raised."""
    bands = _bands_from_table(rows)
    if not bands:
        st.error("Add at least one band with a center guess before fitting.")
        return
    try:
        fit_result = PeakFitter().fit(target_x, target_y, bands)
    except (ValueError, NotImplementedError) as exc:
        st.error(f"Fit failed: {exc}")
        return
    st.session_state["sl_deconv_result"] = fit_result
    st.session_state["sl_deconv_result_file"] = file_name
    st.session_state["sl_deconv_result_digest"] = _bands_digest(rows)


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
    """Deconvolution page: band table (full width, top), target controls (left),
    fit results (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    pipeline_params = st.session_state.get("sl_pipeline_params") or default_pipeline_params()
    with st.spinner("Preparing data…"):
        all_datasets, _errors = get_finals(loaded, pipeline_params)

    if len(loaded) > 1:
        file_name = st.columns([1, 2])[0].selectbox(
            "Select file", list(loaded.keys()), key="deconv_file_select",
        )
    else:
        file_name = next(iter(loaded))
    ds: SpectralDataset = loaded[file_name]["dataset"]

    if ds.measurement_kind != "PL":
        st.info("Deconvolution is available for PL data only (Nanometer/ElectronVolt axes).")
        return

    da_final = final_da(all_datasets.get(file_name))
    if da_final is None:
        st.warning("Processing result not available for this file. Visit the Preprocessing page first.")
        return

    spectral_dim = da_final.dims[-1]
    spatial_dims = [d for d in da_final.dims if d != spectral_dim]

    # The band table spans the full page width (seven numeric columns are unreadable
    # squeezed into the left third), but it needs the display unit and the target
    # spectrum, which the left column below produces. So its slot is reserved here and
    # filled after the left column has run.
    band_slot = st.container()
    left, right = st.columns([1, 2], gap="medium")

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

            # Fitting on this page always runs in energy space, regardless of the
            # file's native storage unit (Nanometer/ElectronVolt) or the Display
            # selector above — those still only affect what's drawn on the chart and
            # which unit the band table is typed in.
            target_x = convert_x(
                target_x, spectral_dim, "energy", laser_nm,
                src_unit=ds.spectral_unit, native_type=ds.spectral_units,
            )

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

    if "deconv_bands_table" not in st.session_state:
        _set_bands(_default_bands_table(target_x))

    units = _BandUnits(spectral_dim, x_unit, laser_nm, ds.spectral_unit)
    with band_slot:
        band_rows, fit_clicked = _render_band_editor(units, file_name)

    with right:
        if fit_clicked:
            _run_fit(target_x, target_y, band_rows, file_name)

        fit_result = st.session_state.get("sl_deconv_result")
        has_fit = fit_result is not None and st.session_state.get("sl_deconv_result_file") == file_name

        fit_title = st.text_input("Chart title", value="Peak Deconvolution", key="deconv_fit_title")
        st.caption(
            "Click anywhere on the plot to drop a new band there — it lands in the table above, "
            "and refits if a fit is already showing."
        )
        # fit_result.x / target_x / band centers are always eV (fitting always runs in
        # energy space — see above), so native_type is forced to ElectronVolt here
        # regardless of ds.spectral_units: these chart builders need to know the
        # *input* array's unit class to correctly re-derive the Display selector's
        # x_unit, and that input is now always eV, not the file's stored unit.
        if has_fit:
            chart_options = make_deconv_fit_echarts(
                fit_result, spectral_dim,
                title=fit_title,
                x_unit=x_unit, laser_nm=laser_nm,
                src_unit=ds.spectral_unit, native_type="ElectronVolt",
            )
        else:
            band_centers_ev = [
                c for row in band_rows
                if (c := _none_or_float(row.get("center_guess"))) is not None
            ]
            chart_options = make_deconv_preview_echarts(
                target_x, target_y, spectral_dim, band_centers_ev,
                title=fit_title,
                x_unit=x_unit, laser_nm=laser_nm,
                src_unit=ds.spectral_unit, native_type="ElectronVolt",
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
            # The click lands in whatever unit the chart is currently displayed in
            # (x_unit); convert it into eV (forcing native_type="ElectronVolt" as the
            # conversion target) since band rows are stored in eV whatever the editor
            # is showing.
            x_ev = units.to_ev(click_value["x"])
            _append_bands([{
                "label": None, "center_guess": round(x_ev, 6),
                "center_min": None, "center_max": None,
                "sigma_guess": None, "sigma_min": None, "sigma_max": None,
            }])
            # Only refit when a fit is already on screen — on a first pass the click is
            # just staging a position, and a surprise fit there hides the preview lines
            # the user is placing.
            if has_fit:
                _run_fit(target_x, target_y, st.session_state["deconv_bands_table"], file_name)
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
            bands = _bands_from_table(band_rows)
            if not bands:
                st.error("Add at least one band with a center guess before fitting.")
            else:
                progress_bar = st.progress(0.0)

                def _cb(done: int, total: int) -> None:
                    progress_bar.progress(done / total)

                # Fit against an eV-coordinate view of da_final (labels only, via
                # assign_coords — no data copy, intensity untouched), matching the
                # single-spectrum fit path above which always runs in energy space.
                da_final_ev = da_final.assign_coords({
                    spectral_dim: convert_x(
                        da_final.coords[spectral_dim].values, spectral_dim, "energy", laser_nm,
                        src_unit=ds.spectral_unit, native_type=ds.spectral_units,
                    )
                })
                with st.spinner("Fitting every pixel…"):
                    batch_result = fit_map_gaussian(da_final_ev, bands, progress_callback=_cb)
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
            c1, c2, c3 = st.columns([1, 1, 2])
            band_label = c1.selectbox("Band", labels, key="deconv_batch_band_select")
            param_name = c2.selectbox(
                "Parameter", ["center", "amplitude", "sigma", "fwhm", "area"],
                key="deconv_batch_param_select",
            )
            with c3:
                colorscale, map_opacity = render_map_display_controls("deconv_map", inline=True)
            z = batch_result.band_results[band_label][param_name]
            row_coords = da_final.coords[spatial_dims[0]].values
            col_coords = da_final.coords[spatial_dims[1]].values
            fig = make_scalar_map_fig(
                z, row_coords, col_coords, ds.image_arr, ds.image_meta,
                cbar_label=f"{band_label} {param_name}",
                title=f"{file_name} — {band_label} {param_name}",
                colorscale=colorscale, map_opacity=map_opacity,
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

"""ECharts chart builders for Progress and Final tabs."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from streamlit_echarts import JsCode

FS_TITLE = 26
FS_AXIS = 22
FS_TICK = 18
FS_LEGEND = 20

COLORS = [
    "#c3121e",
    "#0348a1",
    "#ffb01c",
    "#027608",
    "#1dace6",
    "#9c5300",
    "#9966cc",
    "#ff4500",
]

MULTI_COLORS = [
    "#0348a1",
    "#c3121e",
    "#027608",
    "#ffb01c",
    "#9966cc",
    "#1dace6",
    "#9c5300",
    "#ff4500",
    "#5e4fa2",
    "#3288bd",
    "#66c2a5",
    "#abdda4",
]

VIRIDIS = ["#440154", "#31688e", "#35b779", "#fde725"]
PLASMA_R = ["#f0f921", "#fca636", "#e16462", "#b12a90", "#6a00a8", "#0d0887"]

# Display-only column (spectral-axis) downsampling — analysis/exports always
# use the full-resolution data; this only shrinks what gets serialized into
# a chart's JSON payload (and, with it, what the browser has to render).
MAX_POINTS_PER_TRACE = 1200

# Safety ceiling on how many *spectra* (rows) get their own line in the Final
# chart's index/mean_dev modes. Below this, every spectrum is drawn — no
# silent dropping. Matches the existing >5000-spectra warning in
# preprocessing.py.
MAX_ROWS_DRAWN = 5000

UNIT_LABELS = {
    "wavelength": "wavelength (nm)",
    "energy": "energy (eV)",
    "wavenumber": "wavenumber (cm⁻¹)",
    "raman_shift": "Raman shift (cm⁻¹)",
}


# ---------------------------------------------------------------------------
# Unit conversion (public — used by app.py for range defaults)
# ---------------------------------------------------------------------------


def convert_x(
    x_native: np.ndarray,
    src_dim: str,
    target_unit: str,
    laser_nm: float | None = None,
    *,
    src_unit: str = "",
    native_type: str = "",
) -> np.ndarray:
    """Convert spectral axis from native units to target_unit.

    native_type: canonical WiRE unit string from da.attrs["spectral_units"]
        ("RamanShift", "Wavenumber", "Nanometer", "ElectronVolt").
        When provided, takes precedence over dim-name/src_unit heuristics.
    src_unit: coordinate units string from attrs (e.g. "nm", "1/cm", "cm^-1").
    target_unit: "wavelength" | "energy" | "wavenumber" | "raman_shift"
    """
    nt = native_type.lower().replace("_", "").replace(" ", "")
    u = src_unit.lower()

    # Determine native unit class
    if nt == "ramanshift" or (
        not nt and ("raman" in src_dim.lower() or "shift" in src_dim.lower())
    ):
        native_class = "raman"
    elif nt == "nanometer" or (not nt and ("wavelength" in src_dim.lower() or "nm" in u)):
        native_class = "nm"
    elif nt == "electronvolt" or (not nt and "ev" in u):
        native_class = "ev"
    else:  # Wavenumber, cm^-1, or unknown fallback
        native_class = "wn"

    # RamanShift identity shortcut — no laser needed
    if native_class == "raman" and target_unit == "raman_shift":
        return x_native.astype(float)

    # Convert native → nm
    if native_class == "nm":
        x_nm = x_native.astype(float)
    elif native_class == "ev":
        x_nm = 1239.84 / x_native.astype(float)
    elif native_class == "raman":
        if laser_nm is None:
            raise ValueError("laser_nm is required to convert RamanShift to other units")
        x_nm = 1e7 / (1e7 / laser_nm - x_native.astype(float))
    else:  # wn
        x_nm = 1e7 / x_native.astype(float)

    # Convert nm → target
    if target_unit == "wavelength":
        return x_nm
    if target_unit == "energy":
        return 1239.84 / x_nm
    if target_unit == "wavenumber":
        return 1e7 / x_nm
    if target_unit == "raman_shift":
        if laser_nm is None:
            raise ValueError("laser_nm is required for Raman shift conversion")
        return (1.0 / laser_nm - 1.0 / x_nm) * 1e7
    return x_nm


def convert_x_to_native(
    x_display: float,
    src_dim: str,
    display_unit: str,
    laser_nm: float | None = None,
    *,
    src_unit: str = "",
    native_type: str = "",
) -> float:
    """Inverse of convert_x: a single value in display_unit -> the dataset's native units.

    Used to turn a chart click (reported in display units, since chart series data is
    built via convert_x) back into the native units BandSpec.center_guess expects.
    """
    # display_unit -> nm
    if display_unit == "wavelength":
        x_nm = float(x_display)
    elif display_unit == "energy":
        x_nm = 1239.84 / float(x_display)
    elif display_unit == "wavenumber":
        x_nm = 1e7 / float(x_display)
    elif display_unit == "raman_shift":
        if laser_nm is None:
            raise ValueError("laser_nm is required to convert Raman shift to other units")
        x_nm = 1e7 / (1e7 / laser_nm - float(x_display))
    else:
        x_nm = float(x_display)

    # nm -> native (same native-class detection as convert_x)
    nt = native_type.lower().replace("_", "").replace(" ", "")
    u = src_unit.lower()
    if nt == "ramanshift" or (
        not nt and ("raman" in src_dim.lower() or "shift" in src_dim.lower())
    ):
        if laser_nm is None:
            raise ValueError("laser_nm is required for Raman shift conversion")
        return (1.0 / laser_nm - 1.0 / x_nm) * 1e7
    if nt == "nanometer" or (not nt and ("wavelength" in src_dim.lower() or "nm" in u)):
        return x_nm
    if nt == "electronvolt" or (not nt and "ev" in u):
        return 1239.84 / x_nm
    return 1e7 / x_nm  # Wavenumber, cm^-1, or unknown fallback


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def _top_unit(bottom_unit: str) -> str:
    """Secondary (top) axis: energy unless bottom IS energy → wavelength."""
    return "wavelength" if bottom_unit == "energy" else "energy"


def _top_axis_config(
    x_disp: np.ndarray,
    bottom_unit: str,
    top_unit: str,
    laser_nm: float | None = None,
) -> dict:
    """Return {min, max, inverse} for the secondary top x-axis."""
    if len(x_disp) == 0:
        return {"min": 0, "max": 1, "inverse": False}
    lo, hi = float(x_disp.min()), float(x_disp.max())

    # Convert displayed extremes back to nm for re-conversion to top_unit
    if bottom_unit == "wavelength":
        nm_lo, nm_hi = lo, hi
    elif bottom_unit == "energy":
        nm_lo, nm_hi = 1239.84 / hi, 1239.84 / lo
    elif bottom_unit == "wavenumber":
        nm_lo, nm_hi = 1e7 / hi, 1e7 / lo
    else:  # raman_shift
        if laser_nm:
            d_lo = 1.0 / laser_nm - hi / 1e7
            d_hi = 1.0 / laser_nm - lo / 1e7
            nm_lo = 1.0 / d_lo if d_lo > 0 else laser_nm
            nm_hi = 1.0 / d_hi if d_hi > 0 else laser_nm * 1.5
        else:
            nm_lo, nm_hi = 400.0, 800.0
    nm_lo, nm_hi = min(nm_lo, nm_hi), max(nm_lo, nm_hi)

    if top_unit == "energy":
        e_lo, e_hi = 1239.84 / nm_hi, 1239.84 / nm_lo
        # wavelength bottom → energy decreases as x increases → invert top axis
        inverse = bottom_unit == "wavelength"
        return {"min": round(e_lo, 3), "max": round(e_hi, 3), "inverse": inverse}
    else:  # wavelength
        # energy bottom → nm decreases as x increases → invert top axis
        inverse = bottom_unit == "energy"
        return {"min": round(nm_lo, 1), "max": round(nm_hi, 1), "inverse": inverse}


def _axis_bounds(x_disp: np.ndarray, unit: str) -> tuple[float, float]:
    if len(x_disp) == 0:
        return 0.0, 1.0
    lo, hi = float(x_disp.min()), float(x_disp.max())
    if unit == "energy":
        return math.floor(lo * 100) / 100, math.ceil(hi * 100) / 100
    return math.floor(lo), math.ceil(hi)


def _make_axes(
    x_disp: np.ndarray,
    bottom_unit: str,
    laser_nm: float | None = None,
) -> tuple[dict, dict, dict]:
    """Return (primary_x, secondary_x, y_axis) ECharts axis configs."""
    top_u = _top_unit(bottom_unit)
    top_cfg = _top_axis_config(x_disp, bottom_unit, top_u, laser_nm)
    bmin, bmax = _axis_bounds(x_disp, bottom_unit)

    x_primary = {
        "type": "value",
        "name": UNIT_LABELS[bottom_unit],
        "nameLocation": "middle",
        "nameGap": 35,
        "nameTextStyle": {"fontSize": FS_AXIS},
        "axisLabel": {"fontSize": FS_TICK},
        "min": bmin,
        "max": bmax,
        "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
    }
    x_secondary = {
        "type": "value",
        "name": UNIT_LABELS[top_u],
        "nameLocation": "middle",
        "nameGap": 35,
        "nameTextStyle": {"fontSize": FS_AXIS},
        "axisLabel": {"fontSize": FS_TICK},
        "position": "top",
        "min": top_cfg["min"],
        "max": top_cfg["max"],
        "inverse": top_cfg["inverse"],
        "splitLine": {"show": False},
    }
    y_axis = {
        "type": "value",
        "name": "intensity (a.u.)",
        "nameLocation": "middle",
        "nameGap": 50,
        "nameTextStyle": {"fontSize": FS_AXIS},
        "axisLabel": {"fontSize": FS_TICK},
        "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
    }
    return x_primary, x_secondary, y_axis


# ---------------------------------------------------------------------------
# Tooltip helpers
# ---------------------------------------------------------------------------


def _tooltip_js_parts(bottom_unit: str) -> tuple[str, str]:
    """Return (x_js, secondary_js) snippets for tooltip JavaScript."""
    top_u = _top_unit(bottom_unit)

    if bottom_unit == "wavelength":
        x_js = "x.toFixed(1) + ' nm'"
    elif bottom_unit == "energy":
        x_js = "x.toFixed(3) + ' eV'"
    else:
        x_js = "x.toFixed(1) + ' cm⁻¹'"

    if top_u == "energy":
        if bottom_unit == "wavelength":
            sec_js = "(1239.84 / x).toFixed(3) + ' eV'"
        elif bottom_unit == "raman_shift":
            sec_js = "''"  # shift is a difference — eV conversion is not meaningful
        else:  # wavenumber → eV
            sec_js = "(x / 8065.544).toFixed(3) + ' eV'"
    else:  # top is wavelength (bottom is energy)
        sec_js = "(1239.84 / x).toFixed(1) + ' nm'"

    return x_js, sec_js


def _tooltip_with_ev(bottom_unit: str) -> dict:
    """Axis tooltip: x position in both units + series values."""
    x_js, sec_js = _tooltip_js_parts(bottom_unit)
    js = f"""function(params) {{
    if (!params || !params.length) return '';
    var x = params[0].axisValue;
    var sec = {sec_js};
    var header = sec ? '<b>' + {x_js} + ' &nbsp;/&nbsp; ' + sec + '</b><br/>'
                     : '<b>' + {x_js} + '</b><br/>';
    var html = header;
    params.forEach(function(p) {{
        if (p.seriesType === 'line' && p.seriesName)
            html += p.marker + ' ' + p.seriesName
                  + ':&ensp;<b>' + p.value[1].toExponential(4) + '</b><br/>';
    }});
    return html;
}}"""
    return {"trigger": "axis", "formatter": JsCode(js)}


def _tooltip_x_with_ev(bottom_unit: str, trigger: str = "axis") -> dict:
    """Tooltip showing only x position in both units."""
    x_js, sec_js = _tooltip_js_parts(bottom_unit)
    if trigger == "axis":
        js = f"""function(params) {{
    if (!params || !params.length) return '';
    var x = params[0].axisValue;
    var sec = {sec_js};
    return sec ? '<b>' + {x_js} + ' &nbsp;/&nbsp; ' + sec + '</b>'
               : '<b>' + {x_js} + '</b>';
}}"""
    else:
        js = f"""function(params) {{
    var x = params.data[0];
    var sec = {sec_js};
    return sec ? '<b>' + {x_js} + ' &nbsp;/&nbsp; ' + sec + '</b>'
               : '<b>' + {x_js} + '</b>';
}}"""
    return {"trigger": trigger, "formatter": JsCode(js)}


def _base_title(text: str) -> dict:
    return {"text": text, "top": 5, "textStyle": {"fontSize": FS_TITLE}}


def _base_grid(right: int = 80) -> dict:
    return {"top": 120, "bottom": 155, "left": 80, "right": right}


def _datazoom(
    start_value=None,
    end_value=None,
    x_unit: str = "wavelength",
) -> list:
    # xAxisIndex targets both the primary AND secondary (top, other-unit) axis —
    # every spectral chart here has two x-axes mirroring the same physical range in
    # different units (see _make_axes). Binding the region selector to xAxisIndex 0
    # only would zoom the bottom axis while leaving the top axis frozen at the full
    # original range, so after a zoom the two axes would disagree about what's shown.
    inside = {"type": "inside", "xAxisIndex": [0, 1]}
    slider = {"type": "slider", "xAxisIndex": [0, 1], "bottom": 10, "height": 35}
    if x_unit == "energy":
        fmt_js = "function(v) { return v.toFixed(2); }"
    else:
        fmt_js = "function(v) { return Math.round(v); }"
    slider["labelFormatter"] = JsCode(fmt_js)
    if start_value is not None:
        inside["startValue"] = start_value
        slider["startValue"] = start_value
    if end_value is not None:
        inside["endValue"] = end_value
        slider["endValue"] = end_value
    return [inside, slider]


def _download_toolbox() -> dict:
    return {
        "show": True,
        "top": 5,
        "right": 5,
        "feature": {
            "saveAsImage": {
                "show": True,
                "title": "Download",
                "name": "spectrum",
                "pixelRatio": 4,
                "type": "png",
            }
        },
    }


def _apply_range_mask(
    x_disp: np.ndarray,
    x_range: tuple[float, float] | None,
) -> np.ndarray:
    if x_range is None or len(x_disp) == 0:
        return np.ones(len(x_disp), dtype=bool)
    lo, hi = min(x_range), max(x_range)
    mask = (x_disp >= lo) & (x_disp <= hi)
    return mask if mask.any() else np.ones(len(x_disp), dtype=bool)


def _downsample_cols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the spectral (last) axis to at most MAX_POINTS_PER_TRACE
    points — display only, no-op when there are already that few channels.

    ``y`` may be 1-D (single spectrum) or 2-D ``(n_spectra, n_channels)``;
    the spectral axis is always last. Uses an evenly-spaced index (endpoints
    included) rather than a fitting/binning method — cheap and sufficient
    since ECharts' own "lttb" sampling still runs client-side on top of this
    for the on-screen line.
    """
    n_cols = x.shape[-1]
    if n_cols <= MAX_POINTS_PER_TRACE:
        return x, y
    idx = np.linspace(0, n_cols - 1, MAX_POINTS_PER_TRACE, dtype=int)
    return x[idx], y[..., idx]


def _sample_colorscale(hex_stops: list[str], values: list[float]) -> list[str]:
    """Linearly interpolate hex color stops at normalized float values in [0, 1]."""
    n = len(hex_stops) - 1
    if n == 0:
        return [hex_stops[0]] * len(values)

    def _h2rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    result = []
    for v in values:
        v = max(0.0, min(1.0, float(v)))
        lo = min(int(v * n), n - 1)
        hi = lo + 1
        t = v * n - lo
        r0, g0, b0 = _h2rgb(hex_stops[lo])
        r1, g1, b1 = _h2rgb(hex_stops[hi])
        result.append(
            f"rgb({int(r0 + t * (r1 - r0))},{int(g0 + t * (g1 - g0))},{int(b0 + t * (b1 - b0))})"
        )
    return result


# ---------------------------------------------------------------------------
# Progress / comparison charts
# ---------------------------------------------------------------------------


def make_progress_echarts(
    stages: dict[str, xr.DataArray],
    title: str,
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Line chart of mean spectrum per processing stage."""
    spectral_dim = None
    for da in stages.values():
        spectral_dim = da.dims[-1]
        break
    if spectral_dim is None:
        return {}

    x_native_ref = next(iter(stages.values())).coords[spectral_dim].values
    x_disp_ref = convert_x(
        x_native_ref, spectral_dim, x_unit, laser_nm, src_unit=src_unit, native_type=native_type
    )
    x_primary, x_secondary, y_axis = _make_axes(x_disp_ref, x_unit, laser_nm)

    series = []
    for i, (label, da) in enumerate(stages.items()):
        sd = da.dims[-1]
        xv_native = da.coords[sd].values
        xv_disp = convert_x(
            xv_native, sd, x_unit, laser_nm, src_unit=src_unit, native_type=native_type
        )
        non_spectral = [d for d in da.dims if d != sd]
        mean_spec = da.mean(non_spectral) if non_spectral else da
        yv = mean_spec.values
        xv_disp, yv = _downsample_cols(xv_disp, yv)
        color = COLORS[i % len(COLORS)]
        series.append(
            {
                "type": "line",
                "name": label,
                "color": color,
                "xAxisIndex": 0,
                "data": list(zip(xv_disp.tolist(), yv.tolist())),
                "lineStyle": {"width": 2},
                "symbol": "none",
                "sampling": "lttb",
            }
        )

    return {
        "title": _base_title(title),
        "grid": _base_grid(),
        "xAxis": [x_primary, x_secondary],
        "yAxis": [y_axis],
        "legend": {
            "type": "scroll",
            "orient": "horizontal",
            "bottom": 55,
            "textStyle": {"fontSize": FS_LEGEND},
        },
        "tooltip": _tooltip_with_ev(x_unit),
        "toolbox": _download_toolbox(),
        "dataZoom": _datazoom(
            start_value=x_range[0] if x_range else None,
            end_value=x_range[1] if x_range else None,
            x_unit=x_unit,
        ),
        "animation": False,
        "series": series,
    }


def make_comparison_echarts(
    finals: dict[str, xr.DataArray],
    title: str = "Comparison",
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Overlay the mean spectrum of each file's final DataArray."""
    spectral_dim = None
    for da in finals.values():
        spectral_dim = da.dims[-1]
        break
    if spectral_dim is None:
        return {}

    x_native_ref = next(iter(finals.values())).coords[spectral_dim].values
    x_disp_ref = convert_x(
        x_native_ref, spectral_dim, x_unit, laser_nm, src_unit=src_unit, native_type=native_type
    )
    x_primary, x_secondary, y_axis = _make_axes(x_disp_ref, x_unit, laser_nm)

    series = []
    for i, (name, da) in enumerate(finals.items()):
        sd = da.dims[-1]
        xv_native = da.coords[sd].values
        xv_disp = convert_x(
            xv_native, sd, x_unit, laser_nm, src_unit=src_unit, native_type=native_type
        )
        non_spectral = [d for d in da.dims if d != sd]
        mean_spec = da.mean(non_spectral) if non_spectral else da
        yv = mean_spec.values
        xv_disp, yv = _downsample_cols(xv_disp, yv)
        color = MULTI_COLORS[i % len(MULTI_COLORS)]
        series.append(
            {
                "type": "line",
                "name": name,
                "color": color,
                "xAxisIndex": 0,
                "data": list(zip(xv_disp.tolist(), yv.tolist())),
                "lineStyle": {"width": 2},
                "symbol": "none",
                "sampling": "lttb",
            }
        )

    return {
        "title": _base_title(title),
        "grid": _base_grid(),
        "xAxis": [x_primary, x_secondary],
        "yAxis": [y_axis],
        "legend": {
            "type": "scroll",
            "orient": "horizontal",
            "bottom": 55,
            "textStyle": {"fontSize": FS_LEGEND},
        },
        "tooltip": _tooltip_with_ev(x_unit),
        "toolbox": _download_toolbox(),
        "dataZoom": _datazoom(
            start_value=x_range[0] if x_range else None,
            end_value=x_range[1] if x_range else None,
            x_unit=x_unit,
        ),
        "animation": False,
        "series": series,
    }


# ---------------------------------------------------------------------------
# Final chart (2 modes)
# ---------------------------------------------------------------------------


def make_final_echarts(
    da: xr.DataArray,
    title: str,
    color_by: str = "index",
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
) -> dict:
    """Multi-mode final plot (index, mean_dev)."""
    spectral_dim = da.dims[-1]
    x_native = da.coords[spectral_dim].values
    x_f = convert_x(
        x_native, spectral_dim, x_unit, laser_nm, src_unit=src_unit, native_type=native_type
    )

    x_primary, x_secondary, y_axis = _make_axes(x_f, x_unit, laser_nm)

    if da.ndim == 1:
        x_disp, y_disp = _downsample_cols(x_f, da.values)
        return {
            "title": _base_title(title),
            "grid": _base_grid(),
            "xAxis": [x_primary, x_secondary],
            "yAxis": [{**y_axis, "name": "intensity"}],
            "legend": {"show": False},
            "tooltip": _tooltip_x_with_ev(x_unit, "axis"),
            "toolbox": _download_toolbox(),
            "dataZoom": _datazoom(x_unit=x_unit),
            "animation": False,
            "series": [
                {
                    "type": "line",
                    "xAxisIndex": 0,
                    "data": list(zip(x_disp.tolist(), y_disp.tolist())),
                    "lineStyle": {"color": COLORS[0], "width": 1.5},
                    "symbol": "none",
                    "sampling": "lttb",
                    "name": "spectrum",
                }
            ],
        }

    # Column (spectral-axis) downsampling first — display only, applies
    # before any row handling below so both the NaN-drop and the mean_dev
    # ranking work on the same (already-shrunk) column set.
    spectra = da.values.reshape(-1, da.shape[-1])
    x_s, spectra = _downsample_cols(x_f, spectra)

    spectra_f = spectra
    # Drop dead-pixel rows (all-NaN from CleanData) — NaN is not valid JSON
    # and these rows carry no signal.
    valid_rows = ~np.all(np.isnan(spectra_f), axis=1)
    if not np.all(valid_rows):
        spectra_f = spectra_f[valid_rows]
    n_spectra, n_cols = spectra_f.shape

    # index or mean_dev: one line per spectrum — every spectrum is drawn
    # unless the map is large enough to trip the MAX_ROWS_DRAWN safety
    # ceiling (matches the >5000-spectra warning shown above this chart).
    spectra_s = spectra_f

    if n_spectra > MAX_ROWS_DRAWN:
        idx_sample = np.linspace(0, n_spectra - 1, MAX_ROWS_DRAWN, dtype=int)
    else:
        idx_sample = np.arange(n_spectra)

    alpha = max(0.15, min(0.7, 300 / max(len(idx_sample), 1)))
    y_mid = float(np.nanmean(spectra_s))

    if color_by == "mean_dev":
        mean_s = np.nanmean(spectra_s, axis=0)
        per_spectrum_dev = np.nanmean(np.abs(spectra_s - mean_s), axis=1)
        vmin_v, vmax_v = float(np.nanmin(per_spectrum_dev)), float(np.nanmax(per_spectrum_dev))
        dev_sample = per_spectrum_dev[idx_sample]
        norm_vals = (dev_sample - vmin_v) / (vmax_v - vmin_v + 1e-12)
        palette = _sample_colorscale(PLASMA_R, norm_vals.tolist())
        vm_colors = PLASMA_R
        vm_text = [f"{vmax_v:.3f}", f"{vmin_v:.3f}"]
    else:  # index
        norm_vals = idx_sample.astype(float) / max(n_spectra - 1, 1)
        palette = _sample_colorscale(VIRIDIS, norm_vals.tolist())
        vm_colors = VIRIDIS
        # Colorbar is labelled 1..n to match the 1-based spectrum numbering the
        # rest of the app shows; the colours themselves come from norm_vals.
        vmin_v, vmax_v = 1.0, float(n_spectra)
        vm_text = [str(n_spectra), "1"]

    series: list[dict] = []
    for i, sp_i in enumerate(idx_sample):
        series.append(
            {
                "type": "line",
                "xAxisIndex": 0,
                "data": list(zip(x_s.tolist(), spectra_s[sp_i].tolist())),
                "lineStyle": {"color": palette[i], "width": 1, "opacity": alpha},
                "symbol": "none",
                "sampling": "lttb",
                # Renders the first chunk immediately and streams the rest —
                # keeps first paint + scroll/zoom interaction smooth with
                # hundreds of lines at ~1200 pts each (~960k pts total).
                "progressive": 2000,
                "progressiveThreshold": 100000,
            }
        )

    N = len(series)
    series.append(
        {
            "type": "scatter",
            "xAxisIndex": 0,
            "symbolSize": 0,
            "data": [[float(x_s[0]), y_mid, vmin_v], [float(x_s[0]), y_mid, vmax_v]],
            "silent": True,
        }
    )

    return {
        "title": _base_title(title),
        "grid": _base_grid(right=130),
        "xAxis": [x_primary, x_secondary],
        "yAxis": [{**y_axis, "name": "intensity"}],
        "visualMap": {
            "dimension": 2,
            "seriesIndex": [N],
            "min": vmin_v,
            "max": vmax_v,
            "inRange": {"color": vm_colors},
            "orient": "vertical",
            "right": "2%",
            "top": "10%",
            "bottom": "20%",
            "text": vm_text,
            "textStyle": {"fontSize": 18},
        },
        "legend": {"show": False},
        "tooltip": _tooltip_x_with_ev(x_unit, "axis"),
        "toolbox": _download_toolbox(),
        "dataZoom": _datazoom(x_unit=x_unit),
        "animation": False,
        "series": series,
    }


# ---------------------------------------------------------------------------
# NMF decomposition / peak deconvolution charts
# ---------------------------------------------------------------------------


def make_components_echarts(
    components: np.ndarray,
    x_native: np.ndarray,
    spectral_dim: str,
    title: str = "Component Spectra",
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Overlay NMF component spectra by delegating straight into
    :func:`make_progress_echarts` — each component is wrapped as a 1-D
    DataArray sharing the map's spectral coordinate, so no new
    axis/tooltip/legend building is needed for this chart shape."""
    stages = {
        f"Component {i + 1}": xr.DataArray(
            comp, dims=(spectral_dim,), coords={spectral_dim: x_native}
        )
        for i, comp in enumerate(components)
    }
    return make_progress_echarts(
        stages,
        title,
        x_unit=x_unit,
        laser_nm=laser_nm,
        src_unit=src_unit,
        native_type=native_type,
        x_range=x_range,
    )


def make_nmf_diagnostic_echarts(
    diag: dict,
    title: str = "NMF Diagnostic Curve",
) -> dict:
    """Dual-axis chart: reconstruction error and variance-explained proxy
    vs. component count k — lets the user pick k from where the curves
    elbow, rather than relying on an automatic/hidden choice. Points where
    the fit did not converge within max_iter are drawn as hollow markers
    rather than hidden, since that non-convergence is itself information
    the user should see, not a detail to suppress."""
    k_values = diag["k_values"]
    err = diag["reconstruction_error"]
    frac = diag["fraction_var_explained"]
    converged = diag["converged"]

    err_data = [
        {"value": [k, e], "symbol": "circle" if c else "emptyCircle", "symbolSize": 8 if c else 12}
        for k, e, c in zip(k_values, err, converged)
    ]
    frac_data = [
        {"value": [k, f], "symbol": "circle", "symbolSize": 8} for k, f in zip(k_values, frac)
    ]

    x_axis = {
        "type": "value",
        "name": "n_components (k)",
        "nameLocation": "middle",
        "nameGap": 35,
        "nameTextStyle": {"fontSize": FS_AXIS},
        "axisLabel": {"fontSize": FS_TICK},
        "min": min(k_values),
        "max": max(k_values),
        "minInterval": 1,
        "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
    }
    y_err = {
        "type": "value",
        "name": "reconstruction error",
        "nameLocation": "middle",
        "nameGap": 60,
        "nameTextStyle": {"fontSize": FS_AXIS, "color": COLORS[0]},
        "axisLabel": {"fontSize": FS_TICK, "color": COLORS[0]},
        "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
    }
    y_frac = {
        "type": "value",
        "name": "variance-explained proxy",
        "nameLocation": "middle",
        "nameGap": 50,
        "nameTextStyle": {"fontSize": FS_AXIS, "color": COLORS[1]},
        "axisLabel": {"fontSize": FS_TICK, "color": COLORS[1]},
        "splitLine": {"show": False},
        "min": 0,
        "max": 1,
    }

    tooltip_js = """function(params) {
    if (!params || !params.length) return '';
    var k = params[0].value[0];
    var html = '<b>k = ' + k + '</b><br/>';
    params.forEach(function(p) {
        html += p.marker + ' ' + p.seriesName + ':&ensp;<b>' + p.value[1].toPrecision(4) + '</b><br/>';
    });
    return html;
}"""

    return {
        "title": _base_title(title),
        "grid": _base_grid(right=110),
        "xAxis": [x_axis],
        "yAxis": [y_err, y_frac],
        "legend": {
            "type": "scroll",
            "orient": "horizontal",
            "bottom": 55,
            "textStyle": {"fontSize": FS_LEGEND},
        },
        "tooltip": {"trigger": "axis", "formatter": JsCode(tooltip_js)},
        "toolbox": _download_toolbox(),
        "series": [
            {
                "type": "line",
                "name": "reconstruction error",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": err_data,
                "lineStyle": {"color": COLORS[0], "width": 2},
                "itemStyle": {"color": COLORS[0]},
            },
            {
                "type": "line",
                "name": "variance-explained proxy",
                "xAxisIndex": 0,
                "yAxisIndex": 1,
                "data": frac_data,
                "lineStyle": {"color": COLORS[1], "width": 2},
                "itemStyle": {"color": COLORS[1]},
            },
        ],
    }


def make_mcr_scree_echarts(
    rank: dict,
    title: str = "SVD Scree — how many components?",
) -> dict:
    """Singular-value scree for MCR rank selection.

    Bars are the singular values (how strongly each successive component
    stands out); the line is cumulative variance. The user reads the rank off
    the elbow — the point where bars drop into the flat noise floor — rather
    than an automatic/hidden choice, exactly as with the NMF diagnostic. This
    is a non-spectral chart, so it carries no x-unit selector (CLAUDE.md §5)."""
    svals = list(rank["singular_values"])
    cum = list(rank["cumulative_variance"])
    idx = list(range(1, len(svals) + 1))

    x_axis = {
        "type": "category",
        "data": idx,
        "name": "component #",
        "nameLocation": "middle",
        "nameGap": 35,
        "nameTextStyle": {"fontSize": FS_AXIS},
        "axisLabel": {"fontSize": FS_TICK},
    }
    y_sval = {
        "type": "value",
        "name": "singular value",
        "nameLocation": "middle",
        "nameGap": 70,
        "nameTextStyle": {"fontSize": FS_AXIS, "color": COLORS[0]},
        "axisLabel": {"fontSize": FS_TICK, "color": COLORS[0]},
        "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
    }
    y_cum = {
        "type": "value",
        "name": "cumulative variance",
        "nameLocation": "middle",
        "nameGap": 50,
        "nameTextStyle": {"fontSize": FS_AXIS, "color": COLORS[1]},
        "axisLabel": {"fontSize": FS_TICK, "color": COLORS[1]},
        "splitLine": {"show": False},
        "min": 0,
        "max": 1,
    }

    tooltip_js = """function(params) {
    if (!params || !params.length) return '';
    var html = '<b>component ' + params[0].axisValue + '</b><br/>';
    params.forEach(function(p) {
        html += p.marker + ' ' + p.seriesName + ':&ensp;<b>' + p.value.toPrecision(4) + '</b><br/>';
    });
    return html;
}"""

    return {
        "title": _base_title(title),
        "grid": _base_grid(right=110),
        "xAxis": [x_axis],
        "yAxis": [y_sval, y_cum],
        "legend": {
            "type": "scroll",
            "orient": "horizontal",
            "bottom": 55,
            "textStyle": {"fontSize": FS_LEGEND},
        },
        "tooltip": {"trigger": "axis", "formatter": JsCode(tooltip_js)},
        "toolbox": _download_toolbox(),
        "series": [
            {
                "type": "bar",
                "name": "singular value",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": svals,
                "itemStyle": {"color": COLORS[0]},
            },
            {
                "type": "line",
                "name": "cumulative variance",
                "xAxisIndex": 0,
                "yAxisIndex": 1,
                "data": cum,
                "lineStyle": {"color": COLORS[1], "width": 2},
                "itemStyle": {"color": COLORS[1]},
                "symbolSize": 8,
            },
        ],
    }


def make_mcr_ambiguity_echarts(
    ambiguity: dict,
    title: str = "Rotational Ambiguity (f_max − f_min)",
) -> dict:
    """Per-component ambiguity bars: the width of the feasible band each
    component's relative signal contribution can span while still satisfying
    non-negativity. Near-zero means the component is essentially uniquely
    resolved; a wide bar means several equally-good solutions exist, so its
    exact shape/amplitude should be treated with caution. Non-spectral chart
    (no x-unit selector)."""
    f_range = list(ambiguity["f_range"])
    f_min = list(ambiguity["f_min"])
    f_max = list(ambiguity["f_max"])
    source = ambiguity.get("dominant_source", [""] * len(f_range))
    idx = list(range(1, len(f_range) + 1))

    data = [
        {
            "value": (0.0 if r != r else float(r)),  # NaN -> 0 for display
            "fmin": (None if fm != fm else float(fm)),
            "fmax": (None if fx != fx else float(fx)),
            "src": s,
        }
        for r, fm, fx, s in zip(f_range, f_min, f_max, source)
    ]

    tooltip_js = """function(params) {
    var p = params[0];
    var d = p.data;
    var html = '<b>component ' + p.axisValue + '</b><br/>';
    html += 'ambiguity (f_max − f_min):&ensp;<b>' + (d.value).toPrecision(3) + '</b><br/>';
    if (d.fmin != null) html += 'f_min:&ensp;' + d.fmin.toPrecision(3) + '<br/>';
    if (d.fmax != null) html += 'f_max:&ensp;' + d.fmax.toPrecision(3) + '<br/>';
    if (d.src) html += 'ambiguity in:&ensp;<b>' + d.src + '</b>';
    return html;
}"""

    return {
        "title": _base_title(title),
        "grid": _base_grid(right=40),
        "xAxis": {
            "type": "category",
            "data": idx,
            "name": "component #",
            "nameLocation": "middle",
            "nameGap": 35,
            "nameTextStyle": {"fontSize": FS_AXIS},
            "axisLabel": {"fontSize": FS_TICK},
        },
        "yAxis": {
            "type": "value",
            "name": "f_max − f_min",
            "nameLocation": "middle",
            "nameGap": 60,
            "nameTextStyle": {"fontSize": FS_AXIS},
            "axisLabel": {"fontSize": FS_TICK},
            "splitLine": {"lineStyle": {"color": "#e0e0e0"}},
            "min": 0,
        },
        "tooltip": {"trigger": "axis", "formatter": JsCode(tooltip_js)},
        "toolbox": _download_toolbox(),
        "series": [
            {
                "type": "bar",
                "name": "ambiguity",
                "data": data,
                "itemStyle": {"color": COLORS[2]},
            },
        ],
    }


def make_deconv_fit_echarts(
    fit_result,
    spectral_dim: str,
    title: str = "Peak Deconvolution",
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
) -> dict:
    """Raw spectrum + total fit + per-band dashed sub-curves, with the
    residual on a secondary y-axis (the same dual-y-axis idiom as
    :func:`make_nmf_diagnostic_echarts`) — built from the same
    axis/tooltip/toolbox helpers :func:`make_progress_echarts` uses
    internally, so only the series list and the extra y-axis are new.

    ``fit_result`` is a :class:`peak_fitter.FitResult`.
    """
    x_disp = convert_x(
        fit_result.x,
        spectral_dim,
        x_unit,
        laser_nm,
        src_unit=src_unit,
        native_type=native_type,
    )
    x_primary, x_secondary, y_axis = _make_axes(x_disp, x_unit, laser_nm)

    y_residual = {
        "type": "value",
        "name": "residual",
        "nameLocation": "middle",
        "nameGap": 50,
        "nameTextStyle": {"fontSize": FS_AXIS, "color": "#888"},
        "axisLabel": {"fontSize": FS_TICK, "color": "#888"},
        "splitLine": {"show": False},
    }

    x_list = x_disp.tolist()
    series = [
        {
            "type": "line",
            "name": "data",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "data": list(zip(x_list, fit_result.y_data.tolist())),
            "lineStyle": {"color": "#888888", "width": 1.5},
            "symbol": "none",
        },
        {
            "type": "line",
            "name": "total fit",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "data": list(zip(x_list, fit_result.y_fit.tolist())),
            "lineStyle": {"color": COLORS[0], "width": 2.5},
            "symbol": "none",
        },
        {
            "type": "line",
            "name": "residual",
            "xAxisIndex": 0,
            "yAxisIndex": 1,
            "data": list(zip(x_list, fit_result.residual.tolist())),
            "lineStyle": {"color": "#aaaaaa", "width": 1, "type": "dotted"},
            "symbol": "none",
        },
    ]
    for i, band in enumerate(fit_result.bands):
        color = COLORS[(i + 1) % len(COLORS)]
        series.append(
            {
                "type": "line",
                "name": band.label,
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": list(zip(x_list, band.curve.tolist())),
                "lineStyle": {"color": color, "width": 2, "type": "dashed"},
                "symbol": "none",
            }
        )

    return {
        "title": _base_title(title),
        "grid": _base_grid(),
        "xAxis": [x_primary, x_secondary],
        "yAxis": [y_axis, y_residual],
        "legend": {
            "type": "scroll",
            "orient": "horizontal",
            "bottom": 55,
            "textStyle": {"fontSize": FS_LEGEND},
        },
        "tooltip": _tooltip_with_ev(x_unit),
        "toolbox": _download_toolbox(),
        "dataZoom": _datazoom(x_unit=x_unit),
        "series": series,
    }


def make_deconv_preview_echarts(
    x_native: np.ndarray,
    y: np.ndarray,
    spectral_dim: str,
    band_centers_native: list[float] | None = None,
    title: str = "Peak Deconvolution",
    x_unit: str = "wavelength",
    laser_nm: float | None = None,
    src_unit: str = "",
    native_type: str = "",
) -> dict:
    """Raw target spectrum only (no fit yet) — for picking band positions by eye or
    click before a first fit exists. Dashed vertical markLines show any band centers
    already staged in the band table, built from the same axis/tooltip/toolbox
    helpers make_deconv_fit_echarts uses.
    """
    x_disp = convert_x(
        x_native,
        spectral_dim,
        x_unit,
        laser_nm,
        src_unit=src_unit,
        native_type=native_type,
    )
    x_primary, x_secondary, y_axis = _make_axes(x_disp, x_unit, laser_nm)

    series: list[dict] = [
        {
            "type": "line",
            "name": "data",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "data": list(zip(x_disp.tolist(), y.tolist())),
            "lineStyle": {"color": "#888888", "width": 1.5},
            "symbol": "none",
        }
    ]

    if band_centers_native:
        centers_disp = convert_x(
            np.asarray(band_centers_native, dtype=float),
            spectral_dim,
            x_unit,
            laser_nm,
            src_unit=src_unit,
            native_type=native_type,
        )
        series[0]["markLine"] = {
            "silent": True,
            "symbol": "none",
            "lineStyle": {"color": COLORS[0], "type": "dashed", "width": 1.5},
            "label": {"show": False},
            "data": [{"xAxis": float(c)} for c in centers_disp],
        }

    return {
        "title": _base_title(title),
        "grid": _base_grid(),
        "xAxis": [x_primary, x_secondary],
        "yAxis": [y_axis],
        "legend": {"show": False},
        "tooltip": _tooltip_with_ev(x_unit),
        "toolbox": _download_toolbox(),
        "dataZoom": _datazoom(x_unit=x_unit),
        "series": series,
    }

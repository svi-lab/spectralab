"""Plotly figure builders — map tab only."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from PIL import Image as PILImage

from .charts import UNIT_LABELS, convert_x
from .exclusion import DISPLAY_BASE

FS_TITLE = 26
FS_AXIS  = 22
FS_TICK  = 18
FS_CBAR  = 18


def _colorbar(title: str, tickformat: str = "") -> dict:
    return dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.30,
        yanchor="top",
        thickness=20,
        len=0.9,
        title=dict(text=title, side="bottom", font=dict(size=FS_CBAR)),
        tickfont=dict(size=FS_TICK),
        tickformat=tickformat,
    )


def value_formats(z: np.ndarray) -> tuple[str, str]:
    """``(colorbar_tickformat, hover_format)`` chosen from the map's magnitude.

    One rule for every scalar map (integrated intensity, NMF/MCR abundance,
    fitted band parameters), so a number reads the same way wherever it
    appears. The ladder, keyed on the largest absolute value in the map:

    ============ ==================== =========================
    magnitude    colorbar             example
    ============ ==================== =========================
    < 1          3 decimals           ``0.042``
    1 – 1e3      2 decimals           ``17.50``
    1e3 – 1e6    thousands separator  ``124,730``
    ≥ 1e6        SI prefix, 2 decimals ``1.25M``, ``12.34M``
    ============ ==================== =========================

    Two implementation notes, both forced by how plotly formats numbers:

    * ``exponentformat="SI"`` cannot be combined with an explicit
      ``tickformat`` — plotly.js returns the d3-formatted value before it ever
      looks at the exponent settings. So the SI tier is expressed as d3's own
      ``s`` type instead. ``s`` takes *significant digits*, not decimals, so
      the precision is sized from the value's position within its SI decade
      (``1.25M`` needs 3, ``12.34M`` needs 4, ``123.45M`` needs 5) — that is
      what keeps "2 decimals" true across the whole tier.
    * The hover keeps full digits with separators in the SI tier rather than
      reusing ``s``: hover is a per-pixel readout, and ``s`` would render a
      near-zero pixel on a large map as ``500m`` (milli), which reads as
      nonsense next to a colorbar in ``M``.
    """
    arr = np.asarray(z, dtype=float)
    finite = arr[np.isfinite(arr)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0

    if peak < 1:
        return ".3f", ".3f"
    if peak < 1e3:
        return ".2f", ".2f"
    if peak < 1e6:
        return ",.0f", ",.0f"
    # Digits before the decimal point once the SI prefix is applied: 1 for
    # 1.25M, 2 for 12.34M, 3 for 123.45M — plus the 2 decimals we want.
    mantissa_digits = int(np.floor(np.log10(peak))) % 3 + 1
    return f".{mantissa_digits + 2}s", ",.0f"


@st.cache_data(show_spinner=False, max_entries=8)
def _img_to_b64(arr: np.ndarray) -> str:
    """Convert an RGB numpy array to a grayscale base64-encoded PNG data URI.

    Cached because the same white-light image is re-encoded on every rerun
    that touches the map figure (spectral-range slider, colorscale, etc.)
    even though the image itself essentially never changes.
    """
    img = PILImage.fromarray(arr.astype(np.uint8)).convert("L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_mean_spectrum_option(
    mean_da: xr.DataArray,
    lmin: float,
    lmax: float,
    x_unit: str,
    laser_nm: float | None = None,
    *,
    src_unit: str = "",
    native_type: str = "",
) -> dict:
    """ECharts option for mean spectrum with highlighted selection range.

    ``lmin``/``lmax`` (the highlighted range) must already be in ``x_unit`` — the
    caller is responsible for converting a native-unit range beforehand (they're
    typically the same slider values driving the map's spectral-range selection).
    The plotted x-coordinate is converted from ``mean_da``'s native coordinate to
    ``x_unit`` here via the same :func:`convert_x` every other spectral chart uses.
    """
    spectral_dim = mean_da.dims[0]
    x_native = mean_da.coords[spectral_dim].values
    x = convert_x(
        x_native, spectral_dim, x_unit, laser_nm,
        src_unit=src_unit, native_type=native_type,
    ).tolist()
    y = np.nan_to_num(mean_da.values).tolist()
    return {
        "animation": False,
        "grid": {"top": 24, "bottom": 50, "left": 10, "right": 30},
        "xAxis": {
            "type": "value",
            "name": UNIT_LABELS.get(x_unit, ""),
            "nameLocation": "end",
            "min": min(x) if x else None,
            "max": max(x) if x else None,
            "nameTextStyle": {"fontSize": 12},
            "axisLabel": {"fontSize": 11},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"show": False},
            "name": "",
        },
        "tooltip": {"trigger": "axis"},
        "series": [{
            "type": "line",
            "data": [[xi, yi] for xi, yi in zip(x, y)],
            "showSymbol": False,
            "lineStyle": {"width": 1.5, "color": "#4b8bbe"},
            "markArea": {
                "silent": True,
                "data": [[
                    {"xAxis": lmin, "itemStyle": {"color": "rgba(255, 165, 0, 0.25)"}},
                    {"xAxis": lmax},
                ]],
            },
        }],
    }


def _add_image_overlay(
    fig: go.Figure,
    image_arr: np.ndarray | None,
    image_meta: dict | None,
) -> None:
    """Overlay the white-light image on a heatmap figure, in place."""
    if image_arr is None or image_meta is None:
        return
    ox    = image_meta["origin_x"]
    oy    = image_meta["origin_y"]
    fov_x = image_meta["fov_x"]
    fov_y = image_meta["fov_y"]
    fig.add_layout_image(
        source=_img_to_b64(image_arr),
        xref="x", yref="y",
        x=ox, y=oy,
        sizex=fov_x, sizey=fov_y,
        xanchor="left", yanchor="top",
        opacity=1.0, layer="below", sizing="stretch",
    )


def _apply_map_layout(fig: go.Figure, title: str, cbar_label: str) -> None:
    """Apply the shared title/axes/background layout to a map figure, in place."""
    fig.update_layout(
        title=dict(text=title or cbar_label, font=dict(size=FS_TITLE)),
        xaxis=dict(
            title=dict(text="x (µm)", font=dict(size=FS_AXIS)),
            tickfont=dict(size=FS_TICK),
            showgrid=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(text="y (µm)", font=dict(size=FS_AXIS)),
            tickfont=dict(size=FS_TICK),
            showgrid=False,
            autorange="reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, l=70, r=40, b=160),
        autosize=True,
    )


def make_scalar_map_fig(
    z: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    image_arr: np.ndarray | None,
    image_meta: dict | None,
    *,
    cbar_label: str = "",
    colorscale: str = "Viridis",
    title: str = "",
    value_format: str | None = None,
    map_opacity: float = 0.75,
) -> go.Figure:
    """Heatmap of an arbitrary precomputed 2-D scalar overlaid on the
    white-light image — same rendering as :func:`make_map_fig`, but ``z``
    is supplied directly instead of being derived from a spectral-range
    integration.

    Used for NMF abundance maps and per-pixel fitted Gaussian-band
    parameter maps, where ``z`` already *is* the quantity of interest.

    Parameters
    ----------
    z:
        2-D array, shape ``(n_row, n_col)``, matching ``row_coords`` /
        ``col_coords``.
    row_coords, col_coords:
        Spatial coordinates in µm (e.g. ``da.coords["row"].values``).
    image_arr, image_meta:
        Same as :func:`make_map_fig`.
    value_format:
        Override for the hover number format. ``None`` (the default, and what
        every caller uses) derives both the hover and the colorbar format from
        the data — see :func:`value_formats`. Pass a d3 format string only for
        a quantity that needs a fixed presentation regardless of magnitude.
    map_opacity:
        Opacity of the heatmap trace, drawn above the white-light image.
        Ignored (forced to 1.0) when there is no image to blend with.
    """
    if image_arr is None or image_meta is None:
        map_opacity = 1.0
    tick_fmt, hover_fmt = value_formats(z)
    if value_format is not None:
        hover_fmt = value_format
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=col_coords,
        y=row_coords,
        z=z,
        opacity=map_opacity,
        colorscale=colorscale,
        colorbar=_colorbar(cbar_label, tick_fmt),
        hovertemplate=(
            "x: %{x:.1f} µm<br>y: %{y:.1f} µm<br>"
            + cbar_label.split("(")[0].strip() + f": %{{z:{hover_fmt}}}<extra></extra>"
        ),
        zsmooth=False,
    ))
    _add_image_overlay(fig, image_arr, image_meta)
    _apply_map_layout(fig, title, cbar_label)
    return fig


# Marker colours for the selection map. Kept pixels are near-transparent so
# the intensity heatmap underneath stays readable; excluded ones are opaque.
_SEL_KEPT_RGBA     = "rgba(255,255,255,0.06)"
_SEL_EXCLUDED_RGBA = "rgba(217,48,37,0.95)"    # red   — user-excluded
_SEL_AUTO_RGBA     = "rgba(120,120,120,0.85)"  # grey  — Clean Data removed


def make_selection_map_fig(
    z: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    image_arr: np.ndarray | None,
    image_meta: dict | None,
    excluded_mask: np.ndarray,
    auto_removed_mask: np.ndarray | None = None,
    *,
    colorscale: str = "Viridis",
    title: str = "",
    map_opacity: float = 0.75,
) -> go.Figure:
    """Pixel-pickable map: intensity heatmap + one selectable marker per pixel.

    Hover labels show 1-based row/column (:data:`frontend.exclusion.DISPLAY_BASE`)
    to match the Exclude Spectra tab's index fields.

    ``go.Heatmap`` does not participate in Plotly box/lasso selection, so the
    heatmap is context only (``hoverinfo="skip"``) and a single ``Scattergl``
    trace of pixel centres carries the interaction. Keeping it to *one* trace
    covering *every* pixel is what makes ``point_index`` identical to the
    C-order flat index used everywhere else (see frontend/exclusion.py), so a
    selection maps back to a spectrum without a coordinate lookup.

    Parameters
    ----------
    z:
        2-D ``(n_row, n_col)`` scalar shown underneath (integrated intensity).
    excluded_mask:
        2-D bool, True = manually excluded — drawn red.
    auto_removed_mask:
        2-D bool, True = already NaN before exclusion (Clean Data) — drawn
        grey, so the user can tell automatic from manual removals.
    """
    if image_arr is None or image_meta is None:
        map_opacity = 1.0

    n_row, n_col = int(z.shape[0]), int(z.shape[1])
    excluded = np.asarray(excluded_mask, dtype=bool).ravel()
    auto = (
        np.zeros(n_row * n_col, dtype=bool)
        if auto_removed_mask is None
        else np.asarray(auto_removed_mask, dtype=bool).ravel()
    )

    # Full meshgrid, row-major (y outer, x inner) — the same ravel order as
    # z.reshape(-1) and _shared.scan_geometry, so index i is pixel
    # (i // n_col, i % n_col).
    xs, ys = np.meshgrid(np.asarray(col_coords), np.asarray(row_coords))
    r_idx, c_idx = np.divmod(np.arange(n_row * n_col), n_col)

    colors = np.full(n_row * n_col, _SEL_KEPT_RGBA, dtype=object)
    colors[auto] = _SEL_AUTO_RGBA
    colors[excluded] = _SEL_EXCLUDED_RGBA  # manual wins: it is what's editable

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=col_coords, y=row_coords, z=z,
        opacity=map_opacity, colorscale=colorscale, showscale=False,
        hoverinfo="skip", zsmooth=False,
    ))
    fig.add_trace(go.Scattergl(
        x=xs.ravel(), y=ys.ravel(),
        mode="markers",
        marker=dict(
            size=int(np.clip(600 // max(n_row, n_col, 1), 3, 14)),
            color=colors.tolist(),
            line=dict(width=0),
        ),
        # customdata is the *display* (1-based) pixel address — a hovertemplate
        # cannot do arithmetic, so the offset has to be baked into the data.
        # Readers convert back with ``- DISPLAY_BASE`` (see
        # pages/preprocessing.py::_selected_flat_indices).
        customdata=np.column_stack([r_idx + DISPLAY_BASE, c_idx + DISPLAY_BASE]),
        hovertemplate="row %{customdata[0]}, col %{customdata[1]}<extra></extra>",
        showlegend=False,
        unselected=dict(marker=dict(opacity=1.0)),  # selection must not dim the rest
    ))
    _add_image_overlay(fig, image_arr, image_meta)
    _apply_map_layout(fig, title, "")
    fig.update_layout(dragmode="select", margin=dict(t=80, l=70, r=40, b=60))
    return fig


def make_map_fig(
    da: xr.DataArray,
    image_arr: np.ndarray | None,
    image_meta: dict | None,
    lambda_min: float,
    lambda_max: float,
    quantity: str = "integrated",
    colorscale: str = "Viridis",
    title: str = "",
    spectral_unit: str = "nm",
    map_opacity: float = 0.75,
    label_min: float | None = None,
    label_max: float | None = None,
) -> go.Figure:
    """Heatmap of spectral quantity overlaid on the white-light image.

    Parameters
    ----------
    da:
        Final 3-D DataArray ``(y, x, spectral)``.
    image_arr:
        RGB uint8 numpy array from the WHTL block, or None.
    image_meta:
        Dict with keys ``origin_x``, ``origin_y``, ``fov_x``, ``fov_y``
        (all in µm, origin = top-left of image in stage coordinates).
    lambda_min, lambda_max:
        Spectral integration range, in ``da``'s native coordinate units — always
        used for the actual ``.sel(slice(...))`` integration.
    quantity:
        ``"integrated"`` — sum of intensity in [lambda_min, lambda_max].
        ``"deviation"``  — mean |spectrum − mean_spectrum| in that range.
    colorscale:
        Plotly colorscale name.
    spectral_unit:
        Short unit label for the colorbar caption (e.g. ``"nm"``, ``"eV"``).
    label_min, label_max:
        Range to print in the colorbar caption, in ``spectral_unit`` — may differ
        from ``lambda_min``/``lambda_max`` when the caller displays a different unit
        than ``da``'s native storage unit. Defaults to ``lambda_min``/``lambda_max``
        when omitted (native unit shown == native unit sliced).
    """
    spectral_dim = da.dims[-1]
    x_coords = da.coords["column"].values
    y_coords = da.coords["row"].values

    lmin = min(lambda_min, lambda_max)
    lmax = max(lambda_min, lambda_max)
    da_range = da.sel({spectral_dim: slice(lmin, lmax)})
    spatial_dims = [d for d in da_range.dims if d != spectral_dim]

    label_lo = lmin if label_min is None else min(label_min, label_max)
    label_hi = lmax if label_max is None else max(label_min, label_max)
    _fmt = "{:.2f}" if spectral_unit == "eV" else "{:.0f}"
    label_range = f"{_fmt.format(label_lo)}–{_fmt.format(label_hi)} {spectral_unit}"

    if quantity == "deviation":
        mean_spec = da_range.mean(spatial_dims)
        z = np.abs(da_range - mean_spec).sum(spectral_dim, min_count=1).values
        cbar_label = f"deviation from mean ({label_range})"
    else:
        z = da_range.sum(spectral_dim, min_count=1).values
        cbar_label = f"integrated intensity ({label_range})"

    return make_scalar_map_fig(
        z, y_coords, x_coords, image_arr, image_meta,
        cbar_label=cbar_label, colorscale=colorscale, title=title,
        map_opacity=map_opacity,
    )

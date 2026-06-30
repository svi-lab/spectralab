"""Plotly figure builders — map tab only."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from PIL import Image as PILImage

FS_TITLE = 26
FS_AXIS  = 22
FS_TICK  = 18
FS_CBAR  = 18


def _colorbar(title: str) -> dict:
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
    )


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
    spectral_unit: str,
) -> dict:
    """ECharts option for mean spectrum with highlighted selection range."""
    x = mean_da.coords[mean_da.dims[0]].values.tolist()
    y = np.nan_to_num(mean_da.values).tolist()
    return {
        "animation": False,
        "grid": {"top": 24, "bottom": 50, "left": 10, "right": 30},
        "xAxis": {
            "type": "value",
            "name": "",
            "nameLocation": "end",
            "min": x[0] if x else None,
            "max": x[-1] if x else None,
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
        opacity=0.9, layer="above", sizing="stretch",
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
    value_format: str = ".4g",
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
        Plotly hover number format for ``z``. Defaults to 4 significant
        figures, suitable for quantities of unknown scale (abundances,
        band centers, amplitudes); ``make_map_fig`` passes ``".3f"`` to
        keep its original fixed-decimal display unchanged.
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=col_coords,
        y=row_coords,
        z=z,
        colorscale=colorscale,
        colorbar=_colorbar(cbar_label),
        hovertemplate=(
            "x: %{x:.1f} µm<br>y: %{y:.1f} µm<br>"
            + cbar_label.split("(")[0].strip() + f": %{{z:{value_format}}}<extra></extra>"
        ),
        zsmooth=False,
    ))
    _add_image_overlay(fig, image_arr, image_meta)
    _apply_map_layout(fig, title, cbar_label)
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
        Spectral integration range.
    quantity:
        ``"integrated"`` — sum of intensity in [lambda_min, lambda_max].
        ``"deviation"``  — mean |spectrum − mean_spectrum| in that range.
    colorscale:
        Plotly colorscale name.
    """
    spectral_dim = da.dims[-1]
    x_coords = da.coords["column"].values
    y_coords = da.coords["row"].values

    lmin = min(lambda_min, lambda_max)
    lmax = max(lambda_min, lambda_max)
    da_range = da.sel({spectral_dim: slice(lmin, lmax)})
    spatial_dims = [d for d in da_range.dims if d != spectral_dim]

    if quantity == "deviation":
        mean_spec = da_range.mean(spatial_dims)
        z = np.abs(da_range - mean_spec).sum(spectral_dim, min_count=1).values
        cbar_label = f"deviation from mean ({lmin:.0f}–{lmax:.0f} {spectral_unit})"
    else:
        z = da_range.sum(spectral_dim, min_count=1).values
        cbar_label = f"integrated intensity ({lmin:.0f}–{lmax:.0f} {spectral_unit})"

    return make_scalar_map_fig(
        z, y_coords, x_coords, image_arr, image_meta,
        cbar_label=cbar_label, colorscale=colorscale, title=title,
        value_format=".3f",
    )

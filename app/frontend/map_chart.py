"""Plotly figure builders — map tab only."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from PIL import Image as PILImage

FS_TITLE = 26
FS_AXIS  = 22
FS_TICK  = 18
FS_CBAR  = 18


def _colorbar(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=FS_CBAR)),
        tickfont=dict(size=FS_CBAR),
    )


def _img_to_b64(arr: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64-encoded PNG data URI."""
    buf = BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_map_fig(
    da: xr.DataArray,
    image_arr: np.ndarray | None,
    image_meta: dict | None,
    lambda_min: float,
    lambda_max: float,
    quantity: str = "integrated",
    colorscale: str = "Viridis",
    title: str = "",
    flip_y: bool = False,
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
    flip_y:
        Flip the Y axis if stage Y increases upward instead of downward.
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
        z = np.abs(da_range - mean_spec).sum(spectral_dim).values
        cbar_label = f"deviation from mean ({lmin:.0f}–{lmax:.0f} {spectral_unit})"
    else:
        z = da_range.sum(spectral_dim).values
        cbar_label = f"integrated intensity ({lmin:.0f}–{lmax:.0f} {spectral_unit})"

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_coords,
        y=y_coords,
        z=z,
        colorscale=colorscale,
        colorbar=_colorbar(cbar_label),
        hovertemplate=(
            "x: %{x:.1f} µm<br>y: %{y:.1f} µm<br>"
            + cbar_label.split("(")[0].strip() + ": %{z:.3f}<extra></extra>"
        ),
        zsmooth=False,
    ))

    if image_arr is not None and image_meta is not None:
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
            autorange="normal" if flip_y else "reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, l=70, r=40, b=80),
        autosize=True,
    )
    return fig

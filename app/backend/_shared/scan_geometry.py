# -*- coding: utf-8 -*-
"""Scan footprint geometry extraction for all WDF scan types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from _shared.dataset import SpectralDataset


@dataclass
class ScanGeometry:
    """Scan footprint in stage coordinates (µm).

    ``shape`` determines which fields are meaningful:
      - ``"rect"``   → x_min/x_max/y_min/y_max (padded by ½ step)
      - ``"circle"`` → cx, cy, radius
      - ``"hull"``   → xs, ys (convex-hull vertices, closed)
      - ``"line"``   → xs, ys (ordered path points)
      - ``"points"`` → xs, ys (unordered scatter positions)
    """

    shape: Literal["rect", "circle", "hull", "line", "points"]
    # rect
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    step_x: float = 0.0
    step_y: float = 0.0
    # circle
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    # line / points / hull
    xs: np.ndarray | None = field(default=None, repr=False)
    ys: np.ndarray | None = field(default=None, repr=False)


def _half_step(vals: np.ndarray) -> float:
    """Return ½ the median step size along a sorted coordinate array."""
    if len(vals) < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(vals)))) / 2.0


def _fit_circle_or_hull(xs: np.ndarray, ys: np.ndarray) -> ScanGeometry:
    """Return a circle if the point cloud is nearly circular, else convex hull."""
    cx = float(xs.mean())
    cy = float(ys.mean())
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mean_r = float(r.mean())

    if mean_r > 0 and float(r.std()) / mean_r < 0.15:
        return ScanGeometry(shape="circle", cx=cx, cy=cy, radius=mean_r)

    # Convex hull with scipy; fall back to bounding rect
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([xs, ys])
        hull = ConvexHull(pts)
        vx = xs[hull.vertices]
        vy = ys[hull.vertices]
        # Close the polygon
        vx = np.append(vx, vx[0])
        vy = np.append(vy, vy[0])
        return ScanGeometry(shape="hull", xs=vx, ys=vy)
    except Exception:
        return ScanGeometry(
            shape="rect",
            x_min=float(xs.min()),
            x_max=float(xs.max()),
            y_min=float(ys.min()),
            y_max=float(ys.max()),
        )


def get_scan_geometry(dataset: "SpectralDataset") -> ScanGeometry | None:
    """Extract scan footprint geometry from a SpectralDataset.

    Returns None if no spatial coordinate information is available.
    All coordinates are in stage µm.
    """
    da = dataset.da
    kind: str = da.attrs.get("kind", "")
    data_type: str = da.attrs.get("data_type", "")

    # --- Grid scans (raster_rowmajor, raster_columnmajor, raster_snake grid) ---
    if dataset.is_map and "column" in da.coords and "row" in da.coords:
        x_vals = np.asarray(da.coords["column"].values, dtype=float)
        y_vals = np.asarray(da.coords["row"].values, dtype=float)
        dx = _half_step(x_vals)
        dy = _half_step(y_vals)
        # Full meshgrid in row-major order (y outer, x inner) — must match
        # how da_final.values.reshape(-1, nspec) is laid out for mask alignment.
        xs_grid, ys_grid = np.meshgrid(x_vals, y_vals)
        return ScanGeometry(
            shape="points",
            xs=xs_grid.ravel(),
            ys=ys_grid.ravel(),
            step_x=dx,
            step_y=dy,
        )

    # --- Sequence-type scans: need x/y coords ---
    has_xy = "x" in da.coords and "y" in da.coords
    if not has_xy:
        return None

    xs = np.asarray(da.coords["x"].values, dtype=float)
    ys = np.asarray(da.coords["y"].values, dtype=float)

    # Line scan
    if kind == "line_xy":
        return ScanGeometry(shape="line", xs=xs, ys=ys)

    # Random points
    if kind == "points":
        return ScanGeometry(shape="points", xs=xs, ys=ys)

    # Irregular snake (circle fill etc.)
    if kind == "raster_snake" and data_type == "sequence":
        return _fit_circle_or_hull(xs, ys)

    # Generic fallback for any sequence with spatial coords
    if len(xs) > 1:
        return ScanGeometry(shape="points", xs=xs, ys=ys)

    return None

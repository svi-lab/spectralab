"""Draw scan-footprint overlays on WHTL images using PIL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

if TYPE_CHECKING:
    from _shared.scan_geometry import ScanGeometry


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def stage_to_px(
    x_stage: float | np.ndarray,
    y_stage: float | np.ndarray,
    image_meta: dict,
    width_px: int,
    height_px: int,
    flip_y: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert stage µm coordinates to image pixel coordinates.

    ``image_meta`` must have keys ``origin_x``, ``origin_y``, ``fov_x``, ``fov_y``
    (all µm, origin = top-left of image in stage coords).

    By default stage Y and image Y both increase downward.  Pass ``flip_y=True``
    when stage Y increases upward (matches the Map tab's Flip Y toggle).
    """
    ox = image_meta["origin_x"]
    oy = image_meta["origin_y"]
    fov_x = image_meta["fov_x"]
    fov_y = image_meta["fov_y"]

    px_x = (np.asarray(x_stage, dtype=float) - ox) / fov_x * width_px
    if flip_y:
        px_y = (oy - np.asarray(y_stage, dtype=float)) / fov_y * height_px
    else:
        px_y = (np.asarray(y_stage, dtype=float) - oy) / fov_y * height_px

    return px_x, px_y


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_point_batch(
    draw: ImageDraw.ImageDraw,
    px: np.ndarray,
    py: np.ndarray,
    color: tuple[int, int, int],
    r: int,
    w: int,
    h: int,
) -> None:
    """Draw filled circles at pixel positions (px, py), clipped to image bounds."""
    for ix, iy in zip(px.tolist(), py.tolist()):
        if -r <= ix <= w + r and -r <= iy <= h + r:
            draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=color)


def draw_scan_overlay(
    image_arr: np.ndarray,
    image_meta: dict,
    geometry: ScanGeometry,
    removed_mask: np.ndarray | None = None,
    kept_color: tuple[int, int, int] = (0, 200, 0),
    removed_color: tuple[int, int, int] = (220, 0, 0),
    line_width: int = 3,
    flip_y: bool = False,
) -> np.ndarray:
    """Return a copy of ``image_arr`` (RGB uint8) with the scan footprint drawn.

    Parameters
    ----------
    removed_mask:
        Bool array with the same length as ``geometry.xs``.  ``True`` marks a
        point as removed (drawn in ``removed_color``).  Only meaningful when
        ``geometry.shape == "points"``.  Pass ``None`` to draw all points in
        ``kept_color``.
    kept_color:
        RGB tuple for performed / kept scan positions.  Default green.
    removed_color:
        RGB tuple for CleanData-removed positions.  Default red.
    """
    img = PILImage.fromarray(image_arr.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = image_arr.shape[:2]

    def to_px(xs, ys):
        px, py = stage_to_px(xs, ys, image_meta, w, h, flip_y=flip_y)
        return px.astype(int), py.astype(int)

    s = geometry

    if s.shape == "points" and s.xs is not None:
        # Auto-size dot radius from step sizes, clamped to [1, 8] px
        r = 1
        if s.step_x > 0 and s.step_y > 0:
            r_x = s.step_x / image_meta["fov_x"] * w * 0.45
            r_y = s.step_y / image_meta["fov_y"] * h * 0.45
            r = max(1, min(8, int(min(r_x, r_y))))

        if removed_mask is not None and np.any(removed_mask):
            kept_sel = ~removed_mask
            removed_sel = removed_mask.astype(bool)
            # Draw kept first so removed points render on top
            if np.any(kept_sel):
                px, py = to_px(s.xs[kept_sel], s.ys[kept_sel])
                _draw_point_batch(draw, px, py, kept_color, r, w, h)
            px, py = to_px(s.xs[removed_sel], s.ys[removed_sel])
            _draw_point_batch(draw, px, py, removed_color, r, w, h)
        else:
            px, py = to_px(s.xs, s.ys)
            _draw_point_batch(draw, px, py, kept_color, r, w, h)

    elif s.shape == "circle":
        px_cx, py_cy = stage_to_px(s.cx, s.cy, image_meta, w, h, flip_y=flip_y)
        r_px_x = s.radius / image_meta["fov_x"] * w
        r_px_y = s.radius / image_meta["fov_y"] * h
        bbox = [
            int(px_cx - r_px_x),
            int(py_cy - r_px_y),
            int(px_cx + r_px_x),
            int(py_cy + r_px_y),
        ]
        draw.ellipse(bbox, outline=kept_color, width=line_width)

    elif s.shape in ("line", "hull") and s.xs is not None:
        px, py = to_px(s.xs, s.ys)
        pts = list(zip(px.tolist(), py.tolist()))
        if len(pts) >= 2:
            draw.line(pts, fill=kept_color, width=line_width)

    elif s.shape == "rect":
        xs_rect = np.array([s.x_min, s.x_max, s.x_max, s.x_min, s.x_min])
        ys_rect = np.array([s.y_min, s.y_min, s.y_max, s.y_max, s.y_min])
        px, py = to_px(xs_rect, ys_rect)
        draw.line(list(zip(px.tolist(), py.tolist())), fill=kept_color, width=line_width)

    return np.asarray(img)

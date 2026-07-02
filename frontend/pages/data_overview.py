# -*- coding: utf-8 -*-
"""Data Overview page: file metadata and scan image."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from backend._shared.scan_geometry import ScanGeometry, get_scan_geometry
from backend._shared.scan_overlay import draw_scan_overlay


@st.cache_data(show_spinner=False, max_entries=16)
def _draw_overlay_cached(
    file_hash: str,
    _image_arr: np.ndarray,
    image_meta: dict,
    _geo: ScanGeometry,
) -> np.ndarray:
    return draw_scan_overlay(_image_arr, image_meta, _geo, removed_mask=None)


def _render_file_info(loaded: dict) -> None:
    with st.container(border=True):
        st.markdown('<p class="section-header">File Info</p>', unsafe_allow_html=True)
        if len(loaded) == 1:
            name, entry = next(iter(loaded.items()))
            ds = entry["dataset"]
            lp = ds.laser_power
            et = ds.exposure_time
            comment_line = f"<b>Comment:</b> {ds.comment}<br>" if ds.comment else ""
            st.markdown(
                f'<div class="info-box">'
                f"<b>File:</b> {name}<br>"
                f"<b>Dims:</b> {ds.dims}<br>"
                f"<b>Shape:</b> {ds.shape}<br>"
                f"<b>Ndim:</b> {ds.ndim}<br>"
                f"<b>Kind:</b> {ds.measurement_kind} ({ds.spectral_units or '—'})<br>"
                f"<b>Laser power:</b> {'—' if math.isnan(lp) else f'{lp:.4g}'}<br>"
                f"<b>Exposure time:</b> {'—' if math.isnan(et) else f'{et:.4g}'}<br>"
                f"{comment_line}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{len(loaded)} files loaded.**")
            for name, entry in loaded.items():
                ds = entry["dataset"]
                lp = ds.laser_power
                et = ds.exposure_time
                with st.expander(name, expanded=True):
                    comment_line = f"<b>Comment:</b> {ds.comment}<br>" if ds.comment else ""
                    st.markdown(
                        f'<div class="info-box">'
                        f"<b>Dims:</b> {ds.dims}<br>"
                        f"<b>Shape:</b> {ds.shape}<br>"
                        f"<b>Ndim:</b> {ds.ndim}<br>"
                        f"<b>Kind:</b> {ds.measurement_kind} ({ds.spectral_units or '—'})<br>"
                        f"<b>Laser power:</b> {'—' if math.isnan(lp) else f'{lp:.4g}'}<br>"
                        f"<b>Exposure time:</b> {'—' if math.isnan(et) else f'{et:.4g}'}<br>"
                        f"{comment_line}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def _render_images(loaded: dict) -> None:
    imgs = [
        (name, entry["dataset"], entry["hash"])
        for name, entry in loaded.items()
        if entry["dataset"].image_arr is not None
    ]
    if not imgs:
        st.info("No white-light image found in the uploaded file(s).")
        return
    with st.container(border=True):
        st.markdown('<p class="section-header">Scan Image</p>', unsafe_allow_html=True)
        n_per_row = min(len(imgs), 4)
        for i in range(0, len(imgs), n_per_row):
            batch = imgs[i: i + n_per_row]
            cols = st.columns(len(batch))
            for col, (name, ds, file_hash) in zip(cols, batch):
                if len(loaded) > 1:
                    col.markdown(f"**{name}**")
                arr = ds.image_arr
                geo = get_scan_geometry(ds)
                if geo is not None and ds.image_meta is not None:
                    arr = _draw_overlay_cached(file_hash, arr, ds.image_meta, geo)
                col.image(arr, width="stretch")


def render_data_page() -> None:
    """Data Overview page: file metadata and scan image."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    left, right = st.columns([1, 2], gap="medium")

    with left:
        _render_file_info(loaded)

    with right:
        _render_images(loaded)

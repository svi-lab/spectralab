# -*- coding: utf-8 -*-
"""Left panel: file upload, info display, and pipeline controls."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import streamlit as st

from backend.pipeline import load_wdf
from backend._shared.dataset import SpectralDataset
from .controls import render_clean_data_params, render_crr_params, render_denoising_params


@st.cache_data(show_spinner=False)
def _load_wdf_cached(raw_bytes: bytes) -> SpectralDataset:
    return load_wdf(raw_bytes)


def render_left_panel() -> dict[str, Any]:
    """Render all left-column widgets and return the app state dict.

    Returns
    -------
    dict with keys:
        loaded          – {filename: {"bytes": bytes, "dataset": SpectralDataset}}
        pipeline_params – full processing params dict
        processing_ok   – True when CRR/SC are applicable for this upload set
    """
    st.markdown("## SpectraLab")

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📂 Files</p>', unsafe_allow_html=True)
    if "_uploader_key" not in st.session_state:
        st.session_state["_uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "Drop one or more .wdf files here",
        type=["wdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state['_uploader_key']}",
    )

    if not uploaded_files:
        st.info("Upload one or more .wdf files to get started.")
        st.session_state.pop("_prev_names", None)
        st.stop()

    # Reset processing controls when the uploaded file set changes
    current_names = frozenset(uf.name for uf in uploaded_files)
    if current_names != st.session_state.get("_prev_names"):
        st.session_state["_prev_names"] = current_names
        st.session_state.pop("_file_hashes", None)
        for _key in (
            "cd_enabled", "crr_enabled", "denoise_enabled", "norm_selection", "prog_title",
        ):
            st.session_state.pop(_key, None)

    with st.container(key="remove_files"):
        if st.button("Remove all files", width='stretch'):
            st.session_state["_uploader_key"] += 1
            st.rerun()

    # ── Load files ───────────────────────────────────────────────────────────
    loaded: dict[str, Any] = {}
    load_errors: list[str] = []
    _file_hashes: dict[str, str] = st.session_state.setdefault("_file_hashes", {})

    with st.spinner(f"Reading {len(uploaded_files)} file(s)…"):
        for uf in uploaded_files:
            raw_bytes = uf.read()
            if uf.name not in _file_hashes:
                _file_hashes[uf.name] = hashlib.md5(raw_bytes).hexdigest()
            try:
                dataset = _load_wdf_cached(raw_bytes)
                loaded[uf.name] = {
                    "bytes": raw_bytes,
                    "hash": _file_hashes[uf.name],
                    "dataset": dataset,
                }
            except Exception as exc:
                load_errors.append(f"{uf.name}: {exc}")

    for err in load_errors:
        st.error(f"Failed to read — {err}")
    if not loaded:
        st.stop()

    # ── Per-file validation warnings ─────────────────────────────────────────
    for name, entry in loaded.items():
        ds = entry["dataset"]
        if not ds.is_valid:
            st.warning(f"**{name}**: {ds.validation_msg}")

    if all(not entry["dataset"].is_valid for entry in loaded.values()):
        st.stop()

    # ── Multi-file consistency check ─────────────────────────────────────────
    kinds = {entry["dataset"].measurement_kind for entry in loaded.values()}
    if len(kinds) > 1:
        st.warning(
            f"Mixed measurement types uploaded: {', '.join(sorted(kinds))}. "
            "Processing (CRR / SC) is disabled."
        )
        processing_ok = False
    else:
        processing_ok = next(iter(loaded.values()))["dataset"].preprocessing_available

    # ── File info ────────────────────────────────────────────────────────────
    if len(loaded) == 1:
        name, entry = next(iter(loaded.items()))
        ds = entry["dataset"]
        lp = ds.laser_power
        et = ds.exposure_time
        st.markdown(
            f'<div class="info-box">'
            f"<b>File:</b> {name}<br>"
            f"<b>Dims:</b> {ds.dims}<br>"
            f"<b>Shape:</b> {ds.shape}<br>"
            f"<b>Ndim:</b> {ds.ndim}<br>"
            f"<b>Kind:</b> {ds.measurement_kind} ({ds.spectral_units or '—'})<br>"
            f"<b>Laser power:</b> {'—' if math.isnan(lp) else f'{lp:.4g}'}<br>"
            f"<b>Exposure time:</b> {'—' if math.isnan(et) else f'{et:.4g}'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(loaded)} files loaded.** File details:")
        for name, entry in loaded.items():
            ds = entry["dataset"]
            lp = ds.laser_power
            et = ds.exposure_time
            with st.expander(name, expanded=False):
                st.markdown(
                    f'<div class="info-box">'
                    f"<b>Dims:</b> {ds.dims}<br>"
                    f"<b>Shape:</b> {ds.shape}<br>"
                    f"<b>Ndim:</b> {ds.ndim}<br>"
                    f"<b>Kind:</b> {ds.measurement_kind} ({ds.spectral_units or '—'})<br>"
                    f"<b>Laser power:</b> {'—' if math.isnan(lp) else f'{lp:.4g}'}<br>"
                    f"<b>Exposure time:</b> {'—' if math.isnan(et) else f'{et:.4g}'}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Normalization ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Normalization</p>', unsafe_allow_html=True)
    _NORM_SEGMENTS = ["Before", "After CRR", "After Denoising"]
    norm_selection = st.segmented_control(
        "Normalize at",
        _NORM_SEGMENTS,
        selection_mode="multi",
        key="norm_selection",
        label_visibility="collapsed",
    )
    norm_method: str | None = None
    if norm_selection:
        norm_method = st.selectbox(
            "Method",
            ["min_max", "area"],
            key="norm_method",
            format_func=lambda m: {"min_max": "Min-Max", "area": "Area"}[m],
            help=(
                "**Min-Max** — shifts and scales each spectrum so its minimum "
                "becomes 0 and its maximum becomes 1.  Fast and shape-preserving; "
                "good for comparing peak positions and relative heights when "
                "absolute intensity differences do not matter.  Sensitive to "
                "outlier spikes: a single very high or very low point will "
                "compress the rest of the spectrum.\n\n"
                "**Area** — divides each spectrum by its trapezoidal integral "
                "(area under the curve), then shifts the floor to 0.  Preserves "
                "the relative weight of broad vs. narrow features and is robust "
                "to isolated spikes.  Use this when you want spectra that "
                "represent the same total 'amount' of signal — e.g. before "
                "comparing integrated intensities across samples."
            ),
        )

    st.divider()

    # ── Clean Data ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Clean Data</p>', unsafe_allow_html=True)
    cd_enabled = st.toggle(
        "Remove oversaturated spectra",
        key="cd_enabled",
        help=(
            "Scans every spectrum for **ADC saturation artefacts** — consecutive "
            "channels stuck at exactly 0, which occur when the detector clips to "
            "zero instead of recording the true signal.\n\n"
            "**What it does by data shape:**\n"
            "- **Single spectrum (1D):** issues a warning; spectrum is left unchanged.\n"
            "- **Line scan / series (2D):** drops saturated spectra from the stack "
            "and records which indices were removed.\n"
            "- **Map (3D):** NaN-fills the dead pixels in place, preserving the full "
            "map shape so spatial coordinates stay intact. All downstream steps "
            "(Cosmic Ray Removal, Spectra Cleaner) handle NaN pixels gracefully.\n\n"
            "**When to enable:** if your data contains dead detector pixels or "
            "spectra where the signal went off-scale and clipped to zero. "
            "Run this *before* Cosmic Ray Removal so dead pixels don't interfere "
            "with the spatial reference computation.\n\n"
            "**n_zeros threshold:** how many consecutive zero-valued channels "
            "define a saturated spectrum. The default of 10 avoids false positives "
            "from small gaps while catching real saturation events."
        ),
    )
    cd_params: dict[str, Any] = {}
    if cd_enabled:
        cd_params = render_clean_data_params()

    st.divider()

    # ── CosmicRayRemover ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Cosmic Ray Remover</p>', unsafe_allow_html=True)
    if not processing_ok:
        st.info(
            "CosmicRayRemover and Denoiser require PL data "
            "(Nanometer or ElectronVolt). Not available for this upload."
        )
    crr_enabled = st.toggle(
        "Apply CosmicRayRemover", key="crr_enabled", disabled=not processing_ok
    )
    crr_params: dict[str, Any] = {}
    if crr_enabled:
        crr_params = render_crr_params()

    st.divider()

    # ── Denoising ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Denoising</p>', unsafe_allow_html=True)
    denoise_enabled = st.toggle(
        "Apply Denoiser", key="denoise_enabled", disabled=not processing_ok
    )
    denoise_params: dict[str, Any] = {}
    if denoise_enabled:
        denoise_params = render_denoising_params()

    # ── Assemble pipeline params ─────────────────────────────────────────────
    _ns = norm_selection or []
    _nm = {"method": norm_method} if norm_method else {}
    pipeline_params: dict[str, Any] = {
        "norm1_enabled": "Before" in _ns,
        "norm1":         _nm,
        "cd_enabled":    cd_enabled,
        "cd":            cd_params,
        "crr_enabled":   crr_enabled,
        "crr":           crr_params,
        "norm2_enabled": "After CRR" in _ns,
        "norm2":         _nm,
        "denoise_enabled": denoise_enabled,
        "denoise":         denoise_params,
        "norm3_enabled":   "After Denoising" in _ns,
        "norm3":         _nm,
    }

    return {
        "loaded":          loaded,
        "pipeline_params": pipeline_params,
        "processing_ok":   processing_ok,
    }

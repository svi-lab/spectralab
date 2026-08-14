# -*- coding: utf-8 -*-
"""Shared sidebar: file upload, loading, and validation. Runs on every page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.pipeline import load_wdf
from backend._shared.dataset import SpectralDataset


@st.cache_data(show_spinner=False, max_entries=16)
def _load_wdf_cached(file_id: str, _raw_bytes: bytes) -> SpectralDataset:
    """Keyed on Streamlit's stable per-upload ``file_id`` — excluding the
    bytes from hashing skips an MD5 pass over the full file (~75 MB) on
    every rerun. Trade-off: re-uploading an identical file gets a new
    file_id and re-parses once; accepted."""
    return load_wdf(_raw_bytes)


def render_sidebar() -> bool:
    """Render the shared sidebar: upload, file info, and Remove button.

    Side effects:
        Writes st.session_state["sl_loaded"]         — {fname: {hash, dataset}}
        Writes st.session_state["sl_processing_ok"]  — bool
        Calls st.stop() if files were uploaded but none could be used.

    Returns:
        False if no files have been uploaded yet (caller should render the
        pre-upload landing screen instead of the step bar / page content);
        True otherwise.
    """

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📂 Files</p>', unsafe_allow_html=True)
    if "_sl_uploader_key" not in st.session_state:
        st.session_state["_sl_uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "Drop one or more .wdf files here",
        type=["wdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state['_sl_uploader_key']}",
    )

    if not uploaded_files:
        st.info("Upload one or more .wdf files to get started.")
        st.session_state.pop("sl_loaded", None)
        st.session_state.pop("sl_processing_ok", None)
        return False

    with st.container(key="remove_files"):
        if st.button("Remove all files", width="stretch"):
            st.session_state["_sl_uploader_key"] += 1
            st.session_state.pop("sl_pipeline_params", None)
            st.session_state.pop("sl_sample_structure", None)
            st.session_state.pop("sl_bg_ui", None)
            st.session_state.pop("sl_excluded", None)
            st.session_state.pop("sl_excluded_undo", None)
            st.session_state.pop("_excl_last_selection", None)
            # Finals memo keys include file_id, so stale entries can never be
            # served to a new upload set — but there's no other eviction
            # trigger, so drop them here rather than let them sit resident.
            st.session_state.pop("_sl_finals_memo", None)
            for _key in (
                "cd_enabled", "crr_enabled", "denoise_enabled", "norm_selection", "prog_title",
                "excl_file", "excl_mode", "excl_rows", "excl_cols", "excl_pixels", "excl_flat",
                "bg_enabled", "bg_ref_source", "bg_ref_file",
                "bg_row_min", "bg_row_max", "bg_col_min", "bg_col_max",
                "bg_pt_ratio", "bg_c_override_on", "bg_c_override",
            ):
                st.session_state.pop(_key, None)
            st.rerun()

    # ── Load files ────────────────────────────────────────────────────────────
    # uf.file_id is assigned once by Streamlit when the upload is registered and
    # stays stable across reruns of the same upload — use it as the cache/identity
    # key instead of re-hashing raw_bytes (an MD5 pass) on every single rerun.
    loaded: dict[str, Any] = {}
    load_errors: list[str] = []

    with st.spinner(f"Reading {len(uploaded_files)} file(s)…"):
        for uf in uploaded_files:
            raw_bytes = uf.read()
            try:
                dataset = _load_wdf_cached(uf.file_id, raw_bytes)
                loaded[uf.name] = {
                    "hash": uf.file_id,
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

    # ── Publish to session state ──────────────────────────────────────────────
    st.session_state["sl_loaded"] = loaded
    st.session_state["sl_processing_ok"] = processing_ok
    return True

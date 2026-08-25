"""Shared sidebar: file upload, loading, and validation. Runs on every page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend._shared.dataset import SpectralDataset
from backend.pipeline import load_wdf
from frontend.session import clear_analysis_state


@st.cache_resource(show_spinner=False, max_entries=16)
def _load_wdf_cached(file_id: str, raw_bytes: bytes) -> SpectralDataset:
    """Keyed on Streamlit's stable per-upload ``file_id``.

    ``cache_resource`` returns the same in-memory object (no unpickle copy on
    every rerun). Trade-off: re-uploading an identical file gets a new
    ``file_id`` and re-parses once; accepted."""
    return load_wdf(raw_bytes)


def _upload_matches_loaded(
    uploaded_files: list[Any],
    sl_loaded: dict[str, Any] | None,
) -> bool:
    """True when every uploaded file is already in ``sl_loaded`` with the same id."""
    if not sl_loaded:
        return False
    names = {uf.name for uf in uploaded_files}
    if set(sl_loaded.keys()) != names:
        return False
    for uf in uploaded_files:
        if sl_loaded[uf.name]["hash"] != uf.file_id:
            return False
    return True


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
            clear_analysis_state(st.session_state)
            st.rerun()

    # ── Load files ────────────────────────────────────────────────────────────
    # uf.file_id is assigned once by Streamlit when the upload is registered and
    # stays stable across reruns of the same upload — use it as the cache/identity
    # key instead of re-hashing raw_bytes (an MD5 pass) on every single rerun.
    prev_loaded: dict[str, Any] | None = st.session_state.get("sl_loaded")
    if _upload_matches_loaded(uploaded_files, prev_loaded):
        loaded = prev_loaded
        load_errors: list[str] = []
    else:
        loaded = {}
        load_errors = []
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
            "Cosmic ray removal and denoising are disabled."
        )
        processing_ok = False
    else:
        processing_ok = next(iter(loaded.values()))["dataset"].preprocessing_available

    # ── Publish to session state ──────────────────────────────────────────────
    st.session_state["sl_loaded"] = loaded
    st.session_state["sl_processing_ok"] = processing_ok
    return True

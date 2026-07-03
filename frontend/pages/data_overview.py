# -*- coding: utf-8 -*-
"""Data Overview page: file metadata and scan image."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from backend._shared.scan_geometry import ScanGeometry, get_scan_geometry
from backend._shared.scan_overlay import draw_scan_overlay
from backend.optics import (
    SUBSTRATE_LABELS,
    bare_substrate_summary,
    film_stack_summary,
    lookup_substrate_nk,
)


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


def _fmt_nm(val_nm: float) -> str:
    """Format a length in nm: switch to µm above 10,000 nm; '∞' for inf."""
    if not np.isfinite(val_nm):
        return "∞"
    if val_nm >= 1e4:
        return f"{val_nm / 1e3:.2f} µm"
    return f"{val_nm:.1f} nm"


def _render_sample_structure(loaded: dict) -> None:
    """Bordered 'Sample Structure' card: film inputs + computed optical summary."""
    with st.container(border=True):
        st.markdown('<p class="section-header">Sample Structure</p>', unsafe_allow_html=True)

        ss_store = st.session_state.setdefault("sl_sample_structure", {})

        items = list(loaded.items())
        multi = len(items) > 1

        for name, entry in items:
            ds = entry["dataset"]
            h = entry["hash"]

            if multi:
                expander = st.expander(name, expanded=True)
                ctx = expander
            else:
                ctx = st.container()

            with ctx:
                if ds.laser_nm is None:
                    st.warning("Laser wavelength not found in this file — optical calculations unavailable.")
                    ss_store[name] = {"laser_nm": None, "summary": None}
                    continue

                # ── Sample type ───────────────────────────────────────────
                prev_type = ss_store.get(name, {}).get("sample_type", "film")
                sample_type_label = st.radio(
                    "Sample type",
                    ["Film on substrate", "Bare substrate (reference)"],
                    index=1 if prev_type == "substrate" else 0,
                    key=f"ss_{h}_type",
                    horizontal=True,
                    help=(
                        "Mark bare-substrate files as reference — they get a "
                        "substrate-only optical summary and are offered first as "
                        "the reference in Background Suppression."
                    ),
                )
                is_substrate_only = sample_type_label.startswith("Bare")

                # ── Substrate selector ────────────────────────────────────
                nk_available = lookup_substrate_nk("Si", ds.laser_nm) is not None

                if not nk_available:
                    st.warning(
                        f"No substrate n,k tabulated for λ = {ds.laser_nm:g} nm "
                        "(table covers 355 nm and 320 nm)."
                    )
                    # Force Custom so the user can enter manual values
                    substrate_options = ["Custom"]
                    default_sub_idx = 0
                else:
                    substrate_options = SUBSTRATE_LABELS
                    prev_sub = ss_store.get(name, {}).get("substrate", "Si")
                    default_sub_idx = (
                        substrate_options.index(prev_sub)
                        if prev_sub in substrate_options else 0
                    )

                substrate = st.selectbox(
                    "Substrate",
                    substrate_options,
                    index=default_sub_idx,
                    key=f"ss_{h}_substrate",
                )

                nk = lookup_substrate_nk(substrate, ds.laser_nm) if substrate != "Custom" else None

                if substrate != "Custom" and nk is not None:
                    st.caption(f"n = {nk[0]:.4f},  k = {nk[1]:.2e}  @ {ds.laser_nm:g} nm")
                    sub_n, sub_k = nk
                    manual_sub = False
                else:
                    c1, c2 = st.columns(2)
                    sub_n = c1.number_input(
                        "Substrate n", min_value=1.0, value=float(ss_store.get(name, {}).get("sub_n") or 1.5),
                        step=0.01, format="%.4f", key=f"ss_{h}_sub_n",
                    )
                    sub_k = c2.number_input(
                        "Substrate k", min_value=0.0, value=float(ss_store.get(name, {}).get("sub_k") or 0.0),
                        step=0.001, format="%.4f", key=f"ss_{h}_sub_k",
                    )
                    manual_sub = True

                sub_d_mm = st.number_input(
                    "Substrate thickness (mm)", min_value=0.01,
                    value=float(ss_store.get(name, {}).get("sub_d_mm", 1.0)),
                    step=0.1, format="%.2f", key=f"ss_{h}_sub_d",
                )

                # ── Bare substrate: substrate-only summary, no film ───────
                if is_substrate_only:
                    try:
                        bare = bare_substrate_summary(
                            laser_nm=ds.laser_nm,
                            sub_n=sub_n, sub_k=sub_k, sub_d_mm=sub_d_mm,
                        )
                    except Exception:
                        bare = None

                    ss_store[name] = {
                        "sample_type": "substrate",
                        "substrate": substrate,
                        "sub_n": sub_n, "sub_k": sub_k, "sub_d_mm": sub_d_mm,
                        "laser_nm": ds.laser_nm,
                        "summary": bare,
                    }

                    if bare is not None:
                        st.markdown('<p class="section-header">Optical Summary</p>', unsafe_allow_html=True)
                        alpha_sub = bare["alpha_sub_cm"]
                        delta_sub_str = _fmt_nm(bare["delta_sub_nm"]) if alpha_sub > 0 else "∞ (transparent)"
                        st.markdown(
                            f'<div class="info-box">'
                            f"<b>Bare substrate:</b>  "
                            f"R = {bare['R_air_sub']:.1%}  ·  "
                            f"laser entering = {bare['entry_frac']:.1%}<br>"
                            f"α = {alpha_sub:.3e} cm⁻¹  ·  δ = {delta_sub_str}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Reference file: the fraction entering (1 − R) is the "
                            "denominator of the physics suppression scale c."
                        )
                    continue

                # ── Film inputs ───────────────────────────────────────────
                col_d, col_n, col_k = st.columns(3)
                film_d_nm = col_d.number_input(
                    "Film thickness (nm)", min_value=1.0,
                    value=float(ss_store.get(name, {}).get("film_d_nm", 200.0)),
                    step=10.0, key=f"ss_{h}_film_d",
                )
                film_n = col_n.number_input(
                    "Film n", min_value=1.0,
                    value=float(ss_store.get(name, {}).get("film_n", 2.0)),
                    step=0.01, format="%.4f", key=f"ss_{h}_film_n",
                )
                film_k = col_k.number_input(
                    "Film k", min_value=0.0,
                    value=float(ss_store.get(name, {}).get("film_k", 0.1)),
                    step=0.001, format="%.4f", key=f"ss_{h}_film_k",
                )

                # ── Compute summary ───────────────────────────────────────
                try:
                    summary = film_stack_summary(
                        laser_nm=ds.laser_nm,
                        film_n=film_n, film_k=film_k, film_d_nm=film_d_nm,
                        sub_n=sub_n, sub_k=sub_k, sub_d_mm=sub_d_mm,
                    )
                except Exception:
                    summary = None

                # ── Persist to session state ──────────────────────────────
                ss_store[name] = {
                    "sample_type": "film",
                    "film_d_nm": film_d_nm, "film_n": film_n, "film_k": film_k,
                    "substrate": substrate,
                    "sub_n": sub_n, "sub_k": sub_k, "sub_d_mm": sub_d_mm,
                    "laser_nm": ds.laser_nm,
                    "summary": summary,
                }

                # ── Optical summary display (Tier 3 only) ─────────────────
                if summary is not None:
                    st.markdown('<p class="section-header">Optical Summary</p>', unsafe_allow_html=True)

                    a = summary["alpha_film_cm"]
                    alpha_str = f"{a:.3e}" if a > 0 else "0"
                    delta_str = _fmt_nm(summary["delta_film_nm"])
                    d99s_str  = _fmt_nm(summary["d99_simple_nm"])
                    d99c_str  = _fmt_nm(summary["d99_corrected_nm"])

                    inf_note = "  (k = 0, transparent film)" if film_k == 0 else ""

                    tmm_line = (
                        f"<b>TMM:</b>  "
                        f"R = {summary['tmm_R']:.1%}  ·  "
                        f"A<sub>film</sub> = {summary['tmm_A_film']:.1%}  ·  "
                        f"T→sub = {summary['tmm_T_sub']:.1%}"
                    )
                    bl_line = (
                        f"<b>Beer–Lambert:</b>  "
                        f"R = {summary['bl_R']:.1%}  ·  "
                        f"A<sub>film</sub> = {summary['bl_A_film']:.1%}  ·  "
                        f"T→sub = {summary['bl_T_sub']:.1%}"
                    )
                    alpha_sub = summary["alpha_sub_cm"]
                    delta_sub_str = _fmt_nm(summary["delta_sub_nm"]) if alpha_sub > 0 else "∞ (transparent)"

                    c_line = (
                        f"<b>c<sub>physics</sub></b> = T→sub / (1−R<sub>air-sub</sub>) = "
                        f"{summary['tmm_T_sub']:.3f} / {1.0 - summary['R_air_sub']:.3f} "
                        f"= <b>{summary['c_physics']:.4f}</b>"
                    )
                    st.markdown(
                        f'<div class="info-box">'
                        f"<b>Film:</b>  α = {alpha_str} cm⁻¹  ·  "
                        f"δ (1/e) = {delta_str}{inf_note}  ·  "
                        f"d₉₉ = {d99c_str} (R-corr) / {d99s_str} (simple)<br>"
                        f"{tmm_line}<br>"
                        f"{bl_line}<br>"
                        f"<b>Substrate:</b>  α = {alpha_sub:.3e} cm⁻¹  ·  δ = {delta_sub_str}<br>"
                        f"{c_line}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "TMM accounts for thin-film interference (standing wave); "
                        "Beer–Lambert does not. The difference is largest when "
                        "film thickness ≈ λ/(4n). "
                        "Substrate n,k: Malitson 1965 / Rubin 1985 / Aspnes & Studna 1983."
                    )


def render_data_page() -> None:
    """Data Overview page: file metadata and scan image."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    left, right = st.columns([1, 2], gap="medium")

    with left:
        _render_file_info(loaded)
        _render_sample_structure(loaded)

    with right:
        _render_images(loaded)

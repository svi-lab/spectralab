"""Data page — one block per loaded sample.

Layout
------
``render_data_page`` renders one block per loaded file (multi-file → one
``st.expander`` each, first expanded; single file → one bordered container
titled with the filename). Each block is assembled by ``_render_sample_block``
as a 2×2 card grid:

    ┌ File Info        ┐  ┌ Scan Image ┐
    └ Sample Structure ┘  └ Export     ┘

Files without a white-light image fall back to File Info | Sample Structure
side by side, with Export as a full-width strip below.

Session state
-------------
``sl_sample_structure`` (written here, keyed by file name):
    sample_type : "film" | "substrate" — always written, even with the optics
                  toggle off; orders reference lists on other pages
                  (e.g. Decomposition's fixed-component reference).
    enabled     : bool — the "Calculate optical model" toggle.
    summary     : dict | None — output of ``film_stack_summary`` /
                  ``bare_substrate_summary``. None while the toggle is off.
    laser_nm, substrate, sub_n, sub_k, sub_d_mm, film_d_nm, film_n, film_k
                — last-entered inputs, preserved across toggle off/on.

Caches
------
``_draw_overlay_cached``  — scan-overlay RGB render, keyed on file hash.

Export payloads are deliberately **not** cached and **not** built at render
time: every download button passes a callable (Streamlit's deferred-download
path), so the CSV / .npz bytes are generated only when the user actually
clicks. Building them eagerly kept multi-hundred-MB strings resident in
``st.cache_data`` and re-hashed them on every rerun, which slowed the whole
app down after visiting this page.
"""

from __future__ import annotations

import math
from functools import partial

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

from ..export_utils import (
    mean_spectrum_to_csv,
    mean_spectrum_to_npz,
    spectra_to_csv,
    spectra_to_npz,
)
from ..pipeline_cache import default_pipeline_params, final_da, get_finals

# ───────────────────────────── Cached helpers ─────────────────────────────


@st.cache_data(show_spinner=False, max_entries=16)
def _draw_overlay_cached(
    file_hash: str,
    _image_arr: np.ndarray,
    image_meta: dict,
    _geo: ScanGeometry,
) -> np.ndarray:
    return draw_scan_overlay(_image_arr, image_meta, _geo, removed_mask=None)


# ─────────────────────────────── File info ────────────────────────────────


def _scan_count_label(ds) -> str:
    """Number of spectra with scan geometry, e.g. '2500 (50 × 50 map)'.

    Grid maps use the array shape directly (rows × columns). Other scan
    types are labeled from ``get_scan_geometry``'s ``shape`` field — the
    same classification the scan overlay uses — rather than guessing from
    dimensionality alone, since both line and random-point scans are
    single-dim sequences.
    """
    spatial = [int(s) for d, s in zip(ds.dims, ds.shape) if d != ds.spectral_dim]
    n = int(np.prod(spatial)) if spatial else 1
    if ds.is_map and len(spatial) == 2:
        return f"{n} ({spatial[0]} × {spatial[1]} map)"
    if spatial:
        geo = get_scan_geometry(ds)
        label = {"line": "line", "points": "points"}.get(geo.shape if geo else "", "scan")
        return f"{n} ({label})"
    return "1"


def _render_file_info_card(ds) -> None:
    lp = ds.laser_power
    et = ds.exposure_time
    comment_line = f"<b>Comment:</b> {ds.comment}<br>" if ds.comment else ""
    with st.container(border=True):
        st.markdown('<p class="section-header">File Info</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">'
            f"<b>Scans:</b> {_scan_count_label(ds)}<br>"
            f"<b>Laser power:</b> {'—' if math.isnan(lp) else f'{lp:.4g}'}<br>"
            f"<b>Exposure time:</b> {'—' if math.isnan(et) else f'{et:.4g}'}<br>"
            f"{comment_line}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────── Scan image ───────────────────────────────


def _render_scan_image_card(ds, file_hash: str) -> None:
    """White-light image with the scan-footprint overlay. Caller must ensure
    ``ds.image_arr is not None``."""
    with st.container(border=True):
        st.markdown('<p class="section-header">Scan Image</p>', unsafe_allow_html=True)
        arr = ds.image_arr
        geo = get_scan_geometry(ds)
        if geo is not None and ds.image_meta is not None:
            arr = _draw_overlay_cached(file_hash, arr, ds.image_meta, geo)
        st.image(arr, width="stretch")


# ──────────────────────────── Sample structure ────────────────────────────


def _fmt_nm(val_nm: float) -> str:
    """Format a length in nm: switch to µm above 10,000 nm; '∞' for inf."""
    if not np.isfinite(val_nm):
        return "∞"
    if val_nm >= 1e4:
        return f"{val_nm / 1e3:.2f} µm"
    return f"{val_nm:.1f} nm"


def _render_film_summary(summary: dict, film_k: float) -> None:
    """Light distribution of the excitation beam: where the light goes (reflected /
    absorbed in film / reaches substrate), computed two ways — TMM (with
    thin-film interference) and plain Beer–Lambert."""
    st.markdown(
        '<table class="optics-table">'
        "<tr><th>Light distribution</th><th>TMM</th><th>Beer–Lambert</th></tr>"
        f"<tr><td>Reflected</td>"
        f"<td>{summary['tmm_R']:.1%}</td><td>{summary['bl_R']:.1%}</td></tr>"
        f"<tr><td>Absorbed in film</td>"
        f"<td>{summary['tmm_A_film']:.1%}</td><td>{summary['bl_A_film']:.1%}</td></tr>"
        f"<tr><td>Reaches substrate</td>"
        f"<td>{summary['tmm_T_sub']:.1%}</td><td>{summary['bl_T_sub']:.1%}</td></tr>"
        "</table>",
        unsafe_allow_html=True,
    )
    a = summary["alpha_film_cm"]
    alpha_str = f"{a:.2e}" if a > 0 else "0"
    inf_note = " (k = 0, transparent)" if film_k == 0 else ""
    alpha_sub = summary["alpha_sub_cm"]
    delta_sub_str = _fmt_nm(summary["delta_sub_nm"]) if alpha_sub > 0 else "∞ (transparent)"
    st.caption(
        f"Film: α = {alpha_str} cm⁻¹ · δ = {_fmt_nm(summary['delta_film_nm'])} · "
        f"d₉₉ = {_fmt_nm(summary['d99_corrected_nm'])}{inf_note}  \n"
        f"Substrate: α = {alpha_sub:.2e} cm⁻¹ · δ = {delta_sub_str}"
    )
    st.caption(
        "Same distribution, two methods — TMM includes thin-film interference, "
        "Beer–Lambert does not (difference largest near d ≈ λ/4n)."
    )


def _render_bare_substrate_summary(bare: dict) -> None:
    alpha_sub = bare["alpha_sub_cm"]
    delta_sub_str = _fmt_nm(bare["delta_sub_nm"]) if alpha_sub > 0 else "∞ (transparent)"
    st.markdown(
        f'<div class="info-box">'
        f"Reflected {bare['R_air_sub']:.1%} · entering {bare['entry_frac']:.1%}<br>"
        f"α = {alpha_sub:.2e} cm⁻¹ · δ = {delta_sub_str}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Reference file: the fraction entering (1 − R) is the light that "
        "reaches the substrate after the air interface."
    )


def _render_sample_structure_card(name: str, entry: dict) -> None:
    """Sample-type radio (always visible) + opt-in optical model.

    The toggle gates all optics inputs and computation; when it is off the
    stored ``summary`` is None, but previously entered inputs are preserved so
    re-enabling restores them.
    """
    ds = entry["dataset"]
    h = entry["hash"]
    ss_store = st.session_state.setdefault("sl_sample_structure", {})
    prev = ss_store.get(name, {})

    with st.container(border=True):
        st.markdown('<p class="section-header">Sample Structure</p>', unsafe_allow_html=True)

        # ── Sample type — metadata, independent of the optics toggle ──────
        sample_type_label = st.radio(
            "Sample type",
            ["Film on substrate", "Bare substrate (reference)"],
            index=1 if prev.get("sample_type") == "substrate" else 0,
            key=f"ss_{h}_type",
            horizontal=True,
            help=(
                "Mark bare-substrate files as reference — they are offered "
                "first as the reference on other pages."
            ),
        )
        is_substrate_only = sample_type_label.startswith("Bare")
        sample_type = "substrate" if is_substrate_only else "film"

        calc_enabled = st.toggle(
            "Calculate optical model",
            value=bool(prev.get("enabled", False)),
            key=f"ss_{h}_calc",
            help=(
                "Compute the excitation light distribution (reflection / absorption) "
                "for this stack via TMM and Beer–Lambert."
            ),
        )

        if not calc_enabled:
            ss_store[name] = {
                **prev,
                "sample_type": sample_type,
                "laser_nm": ds.laser_nm,
                "enabled": False,
                "summary": None,
            }
            st.caption("Optics not calculated.")
            return

        if ds.laser_nm is None:
            st.warning(
                "Laser wavelength not found in this file — optical calculations unavailable."
            )
            ss_store[name] = {
                **prev,
                "sample_type": sample_type,
                "laser_nm": None,
                "enabled": True,
                "summary": None,
            }
            return

        # ── Substrate selector ─────────────────────────────────────────────
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
            prev_sub = prev.get("substrate", "Si")
            default_sub_idx = (
                substrate_options.index(prev_sub) if prev_sub in substrate_options else 0
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
        else:
            c1, c2 = st.columns(2)
            sub_n = c1.number_input(
                "Substrate n",
                min_value=1.0,
                value=float(prev.get("sub_n") or 1.5),
                step=0.01,
                format="%.4f",
                key=f"ss_{h}_sub_n",
            )
            sub_k = c2.number_input(
                "Substrate k",
                min_value=0.0,
                value=float(prev.get("sub_k") or 0.0),
                step=0.001,
                format="%.4f",
                key=f"ss_{h}_sub_k",
            )

        sub_d_mm = st.number_input(
            "Substrate thickness (mm)",
            min_value=0.01,
            value=float(prev.get("sub_d_mm", 1.0)),
            step=0.1,
            format="%.2f",
            key=f"ss_{h}_sub_d",
        )

        # ── Bare substrate: substrate-only summary, no film ────────────────
        if is_substrate_only:
            try:
                bare = bare_substrate_summary(
                    laser_nm=ds.laser_nm,
                    sub_n=sub_n,
                    sub_k=sub_k,
                    sub_d_mm=sub_d_mm,
                )
            except Exception:
                bare = None

            ss_store[name] = {
                **prev,
                "sample_type": "substrate",
                "substrate": substrate,
                "sub_n": sub_n,
                "sub_k": sub_k,
                "sub_d_mm": sub_d_mm,
                "laser_nm": ds.laser_nm,
                "enabled": True,
                "summary": bare,
            }
            if bare is not None:
                _render_bare_substrate_summary(bare)
            return

        # ── Film inputs ────────────────────────────────────────────────────
        col_d, col_n, col_k = st.columns(3)
        film_d_nm = col_d.number_input(
            "Film thickness (nm)",
            min_value=1.0,
            value=float(prev.get("film_d_nm", 200.0)),
            step=10.0,
            key=f"ss_{h}_film_d",
        )
        film_n = col_n.number_input(
            "Film n",
            min_value=1.0,
            value=float(prev.get("film_n", 2.0)),
            step=0.01,
            format="%.4f",
            key=f"ss_{h}_film_n",
        )
        film_k = col_k.number_input(
            "Film k",
            min_value=0.0,
            value=float(prev.get("film_k", 0.1)),
            step=0.001,
            format="%.4f",
            key=f"ss_{h}_film_k",
        )

        try:
            summary = film_stack_summary(
                laser_nm=ds.laser_nm,
                film_n=film_n,
                film_k=film_k,
                film_d_nm=film_d_nm,
                sub_n=sub_n,
                sub_k=sub_k,
                sub_d_mm=sub_d_mm,
            )
        except Exception:
            summary = None

        ss_store[name] = {
            **prev,
            "sample_type": "film",
            "film_d_nm": film_d_nm,
            "film_n": film_n,
            "film_k": film_k,
            "substrate": substrate,
            "sub_n": sub_n,
            "sub_k": sub_k,
            "sub_d_mm": sub_d_mm,
            "laser_nm": ds.laser_nm,
            "enabled": True,
            "summary": summary,
        }
        if summary is not None:
            _render_film_summary(summary, film_k)


# ───────────────────────────────── Export ─────────────────────────────────


def _export_stem(name: str) -> str:
    """Strip trailing .wdf for download filenames."""
    lower = name.lower()
    if lower.endswith(".wdf"):
        return name[: -len(".wdf")]
    return name


def _pipeline_export_caption(params: dict | None) -> str:
    if params is None:
        return "No preprocessing set — visit the Preprocessing page first; exporting raw data."
    stages: list[str] = []
    if params.get("norm1_enabled") or params.get("norm2_enabled") or params.get("norm3_enabled"):
        stages.append("normalization")
    if params.get("cd_enabled"):
        stages.append("clean data")
    if params.get("crr_enabled"):
        stages.append("cosmic ray removal")
    if params.get("denoise_enabled"):
        stages.append("denoising")
    if (params.get("excl") or {}).get("masks"):
        stages.append("manual exclusion (excluded spectra blanked, shape preserved)")
    if not stages:
        return "No preprocessing stages enabled — exporting raw data."
    return "Export includes: " + ", ".join(stages) + "."


def _render_export_card(name: str, entry: dict, da_final, params: dict) -> None:
    """Per-sample download buttons; ``da_final`` is None when processing failed."""
    file_hash = entry["hash"]
    with st.container(border=True):
        st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
        if da_final is None:
            st.warning("No exportable data — fix the processing error above.")
            return
        stem = _export_stem(name)
        ds = entry["dataset"]
        axis_label = (
            f"{da_final.dims[-1]} ({ds.spectral_unit})" if ds.spectral_unit else da_final.dims[-1]
        )
        excluded_mask = (params.get("excl") or {}).get("masks", {}).get(name)

        # Payloads are built lazily: st.download_button invokes the callable
        # only when the user clicks, so rendering this card costs nothing.
        # on_click="ignore" also skips the app rerun a click would trigger.
        col_csv, col_npz = st.columns(2)
        with col_csv:
            st.download_button(
                "Full spectra (CSV)",
                partial(spectra_to_csv, da_final, axis_label),
                file_name=f"{stem}.csv",
                mime="text/csv",
                key=f"export_full_csv_{file_hash}",
                on_click="ignore",
                help=(
                    "Plain-text table for Origin / Excel: first column is the "
                    "spectral axis, one column per spectrum (r{row}_c{col}, "
                    "1-based)."
                ),
            )
            if da_final.ndim > 1:
                st.download_button(
                    "Mean spectrum (CSV)",
                    partial(mean_spectrum_to_csv, da_final, axis_label),
                    file_name=f"{stem}_mean.csv",
                    mime="text/csv",
                    key=f"export_mean_csv_{file_hash}",
                    on_click="ignore",
                    help="Two columns: spectral axis, mean intensity over all spectra.",
                )
        with col_npz:
            st.download_button(
                "Full spectra (.npz)",
                partial(spectra_to_npz, da_final, excluded_mask),
                file_name=f"{stem}.npz",
                key=f"export_full_{file_hash}",
                on_click="ignore",
                help="NumPy archive for Python (numpy.load): data + coordinates + exclusion mask.",
            )
            if da_final.ndim > 1:
                st.download_button(
                    "Mean spectrum (.npz)",
                    partial(mean_spectrum_to_npz, da_final),
                    file_name=f"{stem}_mean.npz",
                    key=f"export_mean_{file_hash}",
                    on_click="ignore",
                )
        st.caption(
            "CSV opens directly in Origin / Excel; .npz is for Python. "
            "Files are generated when you click — a full map can take a few seconds."
        )


# ────────────────────────────── Page assembly ─────────────────────────────


def _render_sample_block(name: str, entry: dict, da_final, params: dict) -> None:
    """2×2 card grid for one sample; rebalances when there is no scan image."""
    ds = entry["dataset"]
    h = entry["hash"]
    if ds.image_arr is not None:
        col_l, col_r = st.columns(2, gap="medium")
        with col_l:
            _render_file_info_card(ds)
            _render_sample_structure_card(name, entry)
        with col_r:
            _render_scan_image_card(ds, h)
            _render_export_card(name, entry, da_final, params)
    else:
        col_l, col_r = st.columns(2, gap="medium")
        with col_l:
            _render_file_info_card(ds)
        with col_r:
            _render_sample_structure_card(name, entry)
        _render_export_card(name, entry, da_final, params)


def render_data_page() -> None:
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    stored_params = st.session_state.get("sl_pipeline_params")
    st.caption(_pipeline_export_caption(stored_params))
    params = stored_params or default_pipeline_params()

    all_datasets, errors = get_finals(loaded, params)
    for err in errors:
        st.error(f"Processing error — {err}")

    multi = len(loaded) > 1
    for i, (name, entry) in enumerate(loaded.items()):
        da_final = final_da(all_datasets[name]) if name in all_datasets else None
        if multi:
            with st.expander(name, expanded=(i == 0)):
                _render_sample_block(name, entry, da_final, params)
        else:
            with st.container(border=True):
                st.markdown(f'<p class="section-header">{name}</p>', unsafe_allow_html=True)
                _render_sample_block(name, entry, da_final, params)

"""Preprocessing page — pipeline parameter controls + staged/final charts.

Layout
------
Left column (1/3), top to bottom:

    Quick Setup       — "Per-spectrum preset" / "Collection preset" buttons
                        that enable all three processing steps with
                        per-spectrum or collection-based engines. The
                        per-spectrum path is a valid choice for any data shape
                        (including maps); the collection path needs ≥ 2
                        spectra.
    Normalization     — checkpoint multi-select + method. Selection cascades:
                        a later checkpoint auto-selects every earlier one
                        (After denoising ⇒ After cosmic rays ⇒ Before).
    Processing Steps  — ONE bordered card with four tabs:
                        Clean Data | Cosmic Rays | Denoising |
                        Exclude Spectra.
                        Tabs (not conditional panels) so every stage's widgets
                        render on every rerun and no widget state is lost when
                        switching views. Clean Data and Exclude Spectra each
                        host a small grid (grey = kept, red = auto-removed,
                        orange = manually excluded) rendered into a placeholder
                        after the pipeline has run.

Right column (2/3): ``_render_charts_fragment`` — three chart tabs
(Steps = per-stage progress or multi-file comparison; Final = final
spectra, comparison or single-file; Selection = the interactive pixel picker
that drives manual exclusion).

The Final tab's single-file view carries a **Show** browser
(``_render_browse_controls``): all spectra at once, one spectrum stepped
through with ◀/▶, or a whole map row / column. It only ever ``isel``s the
already-computed final DataArray, so browsing never re-runs the pipeline.

Session state
-------------
``sl_pipeline_params`` (written here every rerun, read by every other page):
    the assembled pipeline params dict passed to ``get_finals``.
``_restore_widget_state`` re-seeds widget keys from it on page entry, since
    st.navigation clears main-content widget state on page transitions.
``sl_excluded`` (frontend/exclusion.py): per-file manual exclusion masks. It
    is a plain session key rather than a widget value, so it survives page
    navigation on its own; ``build_excl_params`` folds it into
    ``sl_pipeline_params["excl"]`` on every rerun.

Caches
------
``_make_final_echarts_cached`` — final-chart ECharts options, keyed on file
hash + pipeline params. ``_make_progress_echarts_cached`` and
``_make_comparison_echarts_cached`` cover the Steps / comparison charts.
Pipeline stage caching itself lives in ``frontend/pipeline_cache.py``; this
page requests ``keep_stages=True`` only while the Steps chart tab is open.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
from backend.cosmic_ray import median_channel_width

from ..charts import convert_x, make_comparison_echarts, make_final_echarts, make_progress_echarts
from ..controls import (
    CRR_ENGINE_1D,
    CRR_ENGINE_2D3D,
    DENOISE_ENGINE_PCA,
    DENOISE_ENGINE_SMOOTHER,
    UNIT_DEFAULT,
    X_UNIT_FMT,
    X_UNIT_OPTIONS,
    ensure_pipeline_widget_defaults,
    render_axis_controls,
    render_clean_data_params,
    render_crr_params,
    render_denoising_params,
    render_map_display_controls,
)
from ..exclusion import (
    DISPLAY_BASE,
    apply_selection,
    build_excl_params,
    clear_mask,
    display_range,
    get_mask,
    has_undo,
    parse_index_spec,
    parse_pixel_spec,
    set_mask,
    spatial_shape,
    to_display,
    undo,
)
from ..map_chart import PLOTLY_CONFIG, make_selection_map_fig
from ..pipeline_cache import final_da, get_finals, stage_dict

# ─────────────────────────── Widget state restore ──────────────────────────


def _restore_widget_state() -> None:
    """Re-seed widget keys from sl_pipeline_params after navigating back here.

    st.navigation clears main-content widget state on page transitions.
    sl_pipeline_params is a plain session-state key, so it survives. We
    pre-seed the widget keys before any widget renders so they pick up the
    saved values rather than their defaults.
    """
    stored = st.session_state.get("sl_pipeline_params")
    if not stored:
        return
    ss = st.session_state

    # ── Normalization ─────────────────────────────────────────────────────
    if "norm_selection" not in ss:
        sel: list[str] = []
        if stored.get("norm1_enabled"):
            sel.append(_NORM_BEFORE)
        if stored.get("norm2_enabled"):
            sel.append(_NORM_AFTER_CRR)
        if stored.get("norm3_enabled"):
            sel.append(_NORM_AFTER_DENOISE)
        ss["norm_selection"] = sel

    if "norm_method" not in ss:
        method = (stored.get("norm1") or stored.get("norm2") or stored.get("norm3") or {}).get(
            "method"
        )
        if method:
            ss["norm_method"] = method

    # ── Clean Data ────────────────────────────────────────────────────────
    if "cd_enabled" not in ss:
        ss["cd_enabled"] = stored.get("cd_enabled", False)
    cd = stored.get("cd") or {}
    if "cd_n_zeros" not in ss and cd.get("n_zeros") is not None:
        ss["cd_n_zeros"] = cd["n_zeros"]

    # ── Cosmic Ray Remover ────────────────────────────────────────────────
    if "crr_enabled" not in ss:
        ss["crr_enabled"] = stored.get("crr_enabled", False)
    crr = stored.get("crr") or {}
    if crr:
        if "crr_engine_mode" not in ss:
            ss["crr_engine_mode"] = CRR_ENGINE_1D if crr.get("force_1d", True) else CRR_ENGINE_2D3D
        for wkey, pkey in (
            ("crr_spike_width", "spike_width"),
            ("crr_broad_spike_width", "broad_spike_width"),
            ("crr_spike_width_nm", "spike_width_units"),
            ("crr_broad_width_nm", "broad_width_units"),
            ("crr_spike_threshold", "spike_threshold"),
            ("crr_spike_passes", "spike_passes"),
            ("crr_map_sensitivity", "map_sensitivity"),
            ("crr_map_disk_radius", "map_disk_radius"),
            ("crr_map_spike_width", "map_spike_width"),
            ("crr_map_method", "map_method"),
            ("crr_map_n_components", "map_n_components"),
        ):
            if wkey not in ss and crr.get(pkey) is not None:
                ss[wkey] = crr[pkey]
        if "crr_consensus_veto" not in ss and crr.get("consensus_veto_fraction") is not None:
            ss["crr_consensus_veto"] = crr["consensus_veto_fraction"] > 0.0

    # ── Denoiser ──────────────────────────────────────────────────────────
    if "denoise_enabled" not in ss:
        ss["denoise_enabled"] = stored.get("denoise_enabled", False)
    den = stored.get("denoise") or {}
    if den:
        if "denoise_engine" not in ss:
            ss["denoise_engine"] = (
                DENOISE_ENGINE_SMOOTHER if den.get("per_spectrum") else DENOISE_ENGINE_PCA
            )
        for wkey, pkey in (
            ("denoise_nc_type", "n_components_type"),
            ("denoise_nc_int", "n_components_int"),
        ):
            if wkey not in ss and pkey in den:
                ss[wkey] = den[pkey]
        if "denoise_baseline" not in ss:
            sub, rst = den.get("subtract_min", True), den.get("restore_min", False)
            ss["denoise_baseline"] = "preserve" if (sub and rst) else ("shape" if sub else "raw")
        sm = den.get("smoother") or {}
        if sm:
            if "denoise_sm_method" not in ss:
                ss["denoise_sm_method"] = sm.get("method", "savgol")
            for wkey, pkey in (
                ("denoise_sm_window_length", "window_length"),
                ("denoise_sm_polyorder", "polyorder"),
                ("denoise_sm_d", "d"),
                ("denoise_sm_auto_lam_calls", "auto_lam_calls"),
                ("denoise_sm_wavelet", "wavelet"),
                ("denoise_sm_wavelet_threshold", "wavelet_threshold"),
            ):
                if wkey not in ss and sm.get(pkey) is not None:
                    ss[wkey] = sm[pkey]
            if "denoise_sm_auto_lam" not in ss:
                ss["denoise_sm_auto_lam"] = sm.get("lam") is None
            if "denoise_sm_lam" not in ss and sm.get("lam") is not None:
                ss["denoise_sm_lam"] = sm["lam"]
            if "denoise_sm_wavelet_level" not in ss:
                lv = sm.get("wavelet_level")
                ss["denoise_sm_wavelet_level"] = 0 if lv is None else lv


# ──────────────────────────── Widget callbacks ─────────────────────────────
# Callbacks run before the script body, so they may freely seed widget keys.


_NORM_BEFORE = "Before processing"
_NORM_AFTER_CRR = "After cosmic rays"
_NORM_AFTER_DENOISE = "After denoising"
_NORM_SEGMENTS = [_NORM_BEFORE, _NORM_AFTER_CRR, _NORM_AFTER_DENOISE]
# Segment values stored by sessions that predate the plain-language rename.
_NORM_LEGACY = {
    "Before": _NORM_BEFORE,
    "After CRR": _NORM_AFTER_CRR,
    "After Denoising": _NORM_AFTER_DENOISE,
}


def _cascade_norm_selection() -> None:
    """Enforce prefix closure on the normalization checkpoints: selecting a
    later checkpoint auto-selects every earlier one."""
    sel = set(st.session_state.get("norm_selection") or [])
    if _NORM_AFTER_DENOISE in sel:
        sel.update((_NORM_BEFORE, _NORM_AFTER_CRR))
    if _NORM_AFTER_CRR in sel:
        sel.add(_NORM_BEFORE)
    st.session_state["norm_selection"] = [s for s in _NORM_SEGMENTS if s in sel]


def _apply_preset(mode: str) -> None:
    """Enable all three processing steps with 1D (per-spectrum) or 3D
    (collection/spatial) engines. Normalization is deliberately untouched;
    every other stage parameter keeps its current value."""
    ss = st.session_state
    ss["cd_enabled"] = True
    ss["crr_enabled"] = True
    ss["denoise_enabled"] = True
    ss["crr_engine_mode"] = CRR_ENGINE_1D if mode == "1d" else CRR_ENGINE_2D3D
    ss["denoise_engine"] = DENOISE_ENGINE_SMOOTHER if mode == "1d" else DENOISE_ENGINE_PCA


# ─────────────────────────── Left column: cards ────────────────────────────


def _render_quick_setup(processing_ok: bool) -> None:
    with st.container(border=True):
        st.markdown('<p class="section-header">Quick Setup</p>', unsafe_allow_html=True)
        col_1d, col_3d = st.columns(2)
        col_1d.button(
            "Per-spectrum preset",
            on_click=_apply_preset,
            args=("1d",),
            disabled=not processing_ok,
            width="stretch",
            help=(
                "Enable every step with per-spectrum engines: Clean Data, "
                "cosmic ray removal (1D) and denoising (Smoother). Each "
                "spectrum is treated independently — works on any data, "
                "including maps and line scans."
            ),
        )
        col_3d.button(
            "Collection preset",
            on_click=_apply_preset,
            args=("3d",),
            disabled=not processing_ok,
            width="stretch",
            help=(
                "Enable every step with collection-based engines: Clean Data, "
                "cosmic ray removal (2D/3D spatial) and denoising (PCA). Uses "
                "the whole set of spectra as statistical context — needs a "
                "map or line scan (at least 2 spectra)."
            ),
        )
        st.caption("Presets enable every processing step; normalization is left as set.")
        if not processing_ok:
            st.caption(
                "Presets are disabled: processing is only available for PL "
                "data with a single measurement type loaded."
            )


def _render_normalization_card() -> tuple[list[str], str | None]:
    """Normalization checkpoints (cascading) + method selector."""
    with st.container(border=True):
        st.markdown('<p class="section-header">Normalization</p>', unsafe_allow_html=True)
        # Map segment values stored by an older session onto the renamed
        # labels — a stale value in widget state would otherwise crash the
        # segmented control.
        if "norm_selection" in st.session_state:
            st.session_state["norm_selection"] = [
                _NORM_LEGACY.get(s, s)
                for s in (st.session_state["norm_selection"] or [])
                if _NORM_LEGACY.get(s, s) in _NORM_SEGMENTS
            ]
        norm_selection = st.segmented_control(
            "Normalize at",
            _NORM_SEGMENTS,
            selection_mode="multi",
            key="norm_selection",
            on_change=_cascade_norm_selection,
            label_visibility="collapsed",
            help=(
                "Where in the pipeline each spectrum gets rescaled. "
                "Selecting a later checkpoint automatically selects the "
                "earlier ones."
            ),
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
    return norm_selection or [], norm_method


def _render_exclusion_tab(loaded: dict[str, Any]) -> Any:
    """Manual exclusion controls. Returns the placeholder for the mask grid.

    Editing writes straight into ``sl_excluded`` (frontend/exclusion.py); the
    caller picks it up via ``build_excl_params`` later in the same rerun, so
    no callback indirection is needed for the typed inputs. The interactive
    map lives in the right column's Selection tab and shares the file choice
    through ``excl_file``.
    """
    st.caption(
        "Drop individual spectra from the analysis. Excluded spectra are "
        "blanked in place — the file keeps its original shape, so exports "
        "and pixel numbering stay valid. Cosmic ray removal and denoising "
        "still see excluded pixels as context, which keeps mask edits "
        "instant."
    )

    names = list(loaded.keys())
    if len(names) > 1:
        fname = st.selectbox("File", names, key="excl_file")
    else:
        fname = names[0]
        st.session_state["excl_file"] = fname

    da_raw = loaded[fname]["dataset"].da
    shape = spatial_shape(da_raw, loaded[fname]["dataset"].spectral_dim)
    if not shape:
        st.info("Single spectra have nothing to exclude.")
        return None

    is_map = len(shape) == 2
    mask = get_mask(fname, shape)

    # `or "Exclude"`: a single-select segmented control returns None when the
    # user clicks the already-active segment (deselect), and there is no
    # meaningful "no mode" state here.
    mode = (
        st.segmented_control(
            "Mode",
            ["Exclude", "Restore"],
            key="excl_mode",
            default="Exclude",
            help="Applies to both the typed indices below and the interactive map.",
        )
        or "Exclude"
    )
    exclude = mode != "Restore"

    if is_map:
        n_row, n_col = shape
        rows_txt = st.text_input(
            "Rows",
            key="excl_rows",
            placeholder="e.g. 1-3, 48",
            help=f"Whole map rows to {mode.lower()}. Valid: {display_range(n_row)}.",
        )
        cols_txt = st.text_input(
            "Columns",
            key="excl_cols",
            placeholder="e.g. 13, 31-34",
            help=f"Whole map columns to {mode.lower()}. Valid: {display_range(n_col)}.",
        )
        pixels_txt = st.text_input(
            "Pixels (row, column)",
            key="excl_pixels",
            placeholder="e.g. (5,8), (10,3)",
        )
        flat_txt = ""
    else:
        n_row, n_col = shape[0], 1
        rows_txt = cols_txt = pixels_txt = ""
        flat_txt = st.text_input(
            "Spectra",
            key="excl_flat",
            placeholder="e.g. 1-4, 8, 11-13",
            help=f"Spectrum indices to {mode.lower()}. Valid: {display_range(shape[0])}.",
        )

    col_apply, col_undo, col_clear = st.columns([2, 1, 1])
    apply_clicked = col_apply.button(mode, key="excl_apply", width="stretch")
    undo_clicked = col_undo.button(
        "Undo",
        key="excl_undo",
        width="stretch",
        disabled=not has_undo(),
    )
    clear_clicked = col_clear.button(
        "Clear",
        key="excl_clear",
        width="stretch",
        disabled=not mask.any(),
        help="Restore every spectrum of this file.",
    )

    if apply_clicked:
        try:
            rows = parse_index_spec(rows_txt, n_row)
            cols = parse_index_spec(cols_txt, n_col)
            pixels = parse_pixel_spec(pixels_txt, n_row, n_col) if pixels_txt.strip() else []
            flat = parse_index_spec(flat_txt, shape[0])
        except ValueError as exc:
            st.error(str(exc))
        else:
            if rows or cols or pixels or flat:
                set_mask(
                    fname,
                    apply_selection(
                        mask,
                        rows=rows,
                        cols=cols,
                        pixels=pixels,
                        flat=flat,
                        exclude=exclude,
                    ),
                )
                st.rerun()
            else:
                st.warning("Nothing to apply — fill in at least one field.")

    if undo_clicked and undo():
        st.rerun()
    if clear_clicked:
        clear_mask(fname, shape)
        st.rerun()

    # Filled by _render_excl_visual once the pipeline has run.
    return st.container()


def _caption_pipeline_order() -> None:
    """One line stating the fixed run order and which steps are currently on.

    The order itself is hard-wired in ``backend.pipeline.run_stage_chain``;
    this is the only place the UI spells it out rather than implying it via
    tab order."""
    ss = st.session_state
    enabled = [
        name
        for key, name in (
            ("cd_enabled", "Clean Data"),
            ("crr_enabled", "Cosmic Rays"),
            ("denoise_enabled", "Denoising"),
        )
        if ss.get(key)
    ]
    order = "Steps run in a fixed order: Clean Data → Cosmic Rays → Denoising → Exclude Spectra"
    if enabled:
        st.caption(f"{order}. Enabled: **{', '.join(enabled)}**.")
    else:
        st.caption(f"{order}. Nothing is enabled yet — charts show raw data.")


def _crr_axis_info(loaded: dict[str, Any]) -> tuple[float | None, str, int]:
    """(channel_width, unit_label, n_spectra) for the CRR widgets.

    Picks the first loaded file as representative — same convention as the
    ``ref_ds`` used for the right-column charts. ``channel_width`` is None
    when nothing is loaded yet or the axis can't be resolved (e.g. a single
    channel), so ``render_crr_params`` falls back to plain channel widgets.
    """
    if not loaded:
        return None, "nm", 1
    ds: SpectralDataset = next(iter(loaded.values()))["dataset"]
    n_spectra = int(np.prod([n for dim, n in zip(ds.dims, ds.shape) if dim != ds.spectral_dim]))
    try:
        channel_width = median_channel_width(ds.da, ds.spectral_dim)
    except ValueError:
        channel_width = None
    return channel_width, ds.spectral_unit, max(n_spectra, 1)


def _render_stage_tabs(
    processing_ok: bool,
    loaded: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    """One bordered card with the four processing steps as tabs.

    Returns (stage_params, cd_visual_slot, excl_visual_slot). Each slot is an
    empty container inside its tab (None when that step is off) that the page
    fills with a removal grid after the pipeline has run.
    """
    _pl_info = "Requires PL data (Nanometer or ElectronVolt). Not available for this upload."
    cd_visual_slot = None

    with st.container(border=True):
        st.markdown('<p class="section-header">Processing Steps</p>', unsafe_allow_html=True)
        _caption_pipeline_order()
        tab_cd, tab_crr, tab_dn, tab_excl = st.tabs(
            ["Clean Data", "Cosmic Rays", "Denoising", "Exclude Spectra"]
        )

        # ── Clean Data ────────────────────────────────────────────────────
        with tab_cd:
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
                    "- **Map (3D):** blanks the dead pixels in place, preserving the full "
                    "map shape so spatial coordinates stay intact. All downstream steps "
                    "(cosmic ray removal, denoising) handle blanked pixels gracefully.\n\n"
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
                if "n_zeros" not in cd_params:
                    cd_params = {
                        "n_zeros": int(st.session_state.get("cd_n_zeros", 10)),
                    }
                # Filled by _render_cd_removed_visual after the pipeline runs.
                cd_visual_slot = st.container()

        # ── Cosmic Rays ───────────────────────────────────────────────────
        with tab_crr:
            if not processing_ok:
                st.info(_pl_info)
            crr_enabled = st.toggle(
                "Remove cosmic ray spikes",
                key="crr_enabled",
                disabled=not processing_ok,
                help=(
                    "Detects the sharp single-channel spikes that cosmic rays "
                    "leave on CCD detectors and repairs them from the "
                    "surrounding signal."
                ),
            )
            crr_params: dict[str, Any] = {}
            if crr_enabled:
                channel_width, unit_label, n_spectra = _crr_axis_info(loaded)
                crr_params = render_crr_params(channel_width, unit_label, n_spectra=n_spectra)

        # ── Denoising ─────────────────────────────────────────────────────
        with tab_dn:
            if not processing_ok:
                st.info(_pl_info)
            denoise_enabled = st.toggle(
                "Reduce noise",
                key="denoise_enabled",
                disabled=not processing_ok,
                help=(
                    "Smooths random detector noise while preserving the "
                    "spectral features — statistically across the whole "
                    "dataset (PCA) or spectrum by spectrum (Smoother)."
                ),
            )
            denoise_params: dict[str, Any] = {}
            if denoise_enabled:
                denoise_params = render_denoising_params()

        # ── Exclude Spectra ───────────────────────────────────────────────
        with tab_excl:
            excl_visual_slot = _render_exclusion_tab(loaded)

    stage_params = {
        "cd_enabled": cd_enabled,
        "cd": cd_params,
        "crr_enabled": crr_enabled,
        "crr": crr_params,
        "denoise_enabled": denoise_enabled,
        "denoise": denoise_params,
    }
    return stage_params, cd_visual_slot, excl_visual_slot


def _render_preprocessing_params(
    processing_ok: bool,
    loaded: dict,
) -> tuple[dict[str, Any], Any, Any]:
    """Render the left-column cards; return (pipeline_params, cd_slot, excl_slot)."""
    _render_quick_setup(processing_ok)
    norm_selection, norm_method = _render_normalization_card()
    stage_params, cd_visual_slot, excl_visual_slot = _render_stage_tabs(processing_ok, loaded)

    _nm = {"method": norm_method} if norm_method else {}
    pipeline_params = {
        "norm1_enabled": _NORM_BEFORE in norm_selection,
        "norm1": _nm,
        "norm2_enabled": _NORM_AFTER_CRR in norm_selection,
        "norm2": _nm,
        "norm3_enabled": _NORM_AFTER_DENOISE in norm_selection,
        "norm3": _nm,
        **stage_params,
        # Read from sl_excluded, which the Exclude tab and the Selection chart
        # have already updated by this point in the rerun.
        "excl": build_excl_params(loaded),
    }
    return pipeline_params, cd_visual_slot, excl_visual_slot


# ───────────────────── Clean Data removed-spectra visual ───────────────────


_CD_KEPT_RGB = np.array([232, 234, 237], dtype=np.uint8)  # 0 — light grey, kept
_CD_REMOVED_RGB = np.array([217, 48, 37], dtype=np.uint8)  # 1 — red, auto-removed
_CD_EXCLUDED_RGB = np.array([245, 158, 11], dtype=np.uint8)  # 2 — orange, user-excluded
_CD_PALETTE = np.stack([_CD_KEPT_RGB, _CD_REMOVED_RGB, _CD_EXCLUDED_RGB])
_CD_GRID_MAX_PX = 360  # target image width in px
_CD_WRAP_COLS = 50  # line scans wrap into rows this wide
_CD_MAX_LISTED = 12  # list removed indices explicitly up to this many


def _cd_removed_mask(da, spectral_dim: str) -> np.ndarray:
    """Bool mask of removed spectra (all-NaN along the spectral axis).

    Works for both shapes the pipeline produces: 2-D line scans (removed rows
    NaN-padded back by ``stage_clean_data``'s reindex) and 3-D maps (pixels
    NaN-filled in place by ``CleanData``).
    """
    if da.dims[-1] != spectral_dim:
        spatial = [d for d in da.dims if d != spectral_dim]
        da = da.transpose(*spatial, spectral_dim)
    return np.all(np.isnan(da.values), axis=-1)


def _removal_grid_rgb(categories: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Small RGB indicator image from a 2-D category grid.

    ``categories`` is 0 = kept, 1 = auto-removed (Clean Data), 2 = manually
    excluded — or a plain bool mask, which maps to kept/auto-removed so the
    Clean Data callers stay unchanged.

    ``valid`` marks real cells (padding from line-scan wrapping renders
    white). Cells are upscaled with ``np.kron`` to stay readable; thin white
    grid lines separate cells when they are large enough.
    """
    cat = np.asarray(categories)
    if cat.dtype == bool:
        cat = cat.astype(np.uint8)
    rgb = _CD_PALETTE[cat]
    if valid is not None:
        rgb = rgb.copy()
        rgb[~valid] = 255
    nx = cat.shape[1]
    cell = int(np.clip(_CD_GRID_MAX_PX // max(nx, 1), 2, 14))
    img = np.kron(rgb, np.ones((cell, cell, 1), dtype=np.uint8))
    if cell >= 4:
        img[cell - 1 :: cell, :, :] = 255
        img[:, cell - 1 :: cell, :] = 255
    return img


def _wrap_line_mask(mask: np.ndarray, ncols: int = _CD_WRAP_COLS) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a 1-D removal mask into a (nrows, ncols) grid + validity mask."""
    n = mask.size
    ncols = min(n, ncols)
    nrows = -(-n // ncols)  # ceil division
    padded = np.zeros(nrows * ncols, dtype=bool)
    padded[:n] = mask
    valid = np.zeros(nrows * ncols, dtype=bool)
    valid[:n] = True
    return padded.reshape(nrows, ncols), valid.reshape(nrows, ncols)


def _render_cd_removed_visual(slot, all_datasets: dict, loaded: dict[str, Any]) -> None:
    """Fill the Clean Data tab's placeholder with per-file removal grids.

    Reads the ``clean_data`` stage variable out of the pipeline result (retained
    even when ``keep_stages=False``), so it always reflects exactly what the
    current parameters removed.
    """
    multi = len(all_datasets) > 1
    with slot:
        for name, stage_ds in all_datasets.items():
            if "clean_data" not in stage_ds.data_vars:
                continue
            da = stage_ds["clean_data"]
            label = f"**{name}** — " if multi else ""

            if da.ndim == 1:
                treated = "Oversaturation Check" in (da.attrs.get("treatments") or {})
                st.caption(
                    f"{label}oversaturated spectrum detected — warning only, "
                    "single spectra are not removed."
                    if treated
                    else f"{label}no oversaturated spectra detected."
                )
                continue

            mask = _cd_removed_mask(da, loaded[name]["dataset"].spectral_dim)
            n_removed, total = int(mask.sum()), int(mask.size)
            if n_removed == 0:
                st.caption(f"{label}no oversaturated spectra detected.")
                continue

            if mask.ndim == 1:
                # Listed in display numbering — these are the same indices the
                # Exclude Spectra tab's "Spectra" field takes.
                idx = np.flatnonzero(mask) + DISPLAY_BASE
                listed = ", ".join(map(str, idx[:_CD_MAX_LISTED]))
                idx_note = f" (indices {listed}{'…' if idx.size > _CD_MAX_LISTED else ''})"
                grid, valid = _wrap_line_mask(mask)
                st.image(_removal_grid_rgb(grid, valid))
                st.caption(
                    f"{label}removed {n_removed} / {total} spectra{idx_note}. "
                    "Grey = kept, red = removed; index runs left→right, top→bottom."
                )
            else:
                st.image(_removal_grid_rgb(mask))
                st.caption(
                    f"{label}{n_removed} / {total} map pixels removed "
                    "(grey = kept, red = removed; map layout)."
                )


def _excl_categories(user_mask: np.ndarray, final_nan: np.ndarray) -> np.ndarray:
    """Category grid for the exclusion readout: 0 kept, 1 auto-removed, 2 excluded.

    ``final_nan`` is every all-NaN spectrum in the final result, which is the
    union of Clean Data's removals and the user's — so anything NaN that the
    user did not pick came from an automatic stage.
    """
    cat = np.zeros(user_mask.shape, dtype=np.uint8)
    cat[final_nan & ~user_mask] = 1
    cat[user_mask] = 2
    return cat


def _render_excl_visual(slot, all_datasets: dict, loaded: dict[str, Any]) -> None:
    """Fill the Exclude tab's placeholder with the mask grid for the chosen file."""
    fname = st.session_state.get("excl_file")
    if fname not in all_datasets:
        return
    da_final = final_da(all_datasets[fname])
    spectral_dim = loaded[fname]["dataset"].spectral_dim
    shape = spatial_shape(da_final, spectral_dim)
    if not shape:
        return

    user_mask = get_mask(fname, shape)
    final_nan = _cd_removed_mask(da_final, spectral_dim)
    n_user = int(user_mask.sum())
    n_auto = int((final_nan & ~user_mask).sum())
    total = int(user_mask.size)

    with slot:
        cat = _excl_categories(user_mask, final_nan)
        if cat.ndim == 1:
            grid, valid = _wrap_line_mask(cat.astype(bool))
            # _wrap_line_mask pads a bool mask; re-pad the categories the same way.
            padded = np.zeros(grid.size, dtype=np.uint8)
            padded[: cat.size] = cat
            st.image(_removal_grid_rgb(padded.reshape(grid.shape), valid))
            st.caption("Index runs left→right, top→bottom.")
        else:
            st.image(_removal_grid_rgb(cat))

        parts = [f"**{n_user}** excluded"]
        if n_auto:
            parts.append(f"{n_auto} auto-removed")
        st.caption(
            f"{' · '.join(parts)} / {total} spectra. "
            "Grey = kept · red = auto-removed (Clean Data) · orange = manually excluded."
        )


# ───────────────────────── Right column: chart tabs ────────────────────────


def _run_preprocessing(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    *,
    keep_stages: bool = False,
) -> tuple[dict, list[str]]:
    with st.spinner(f"Processing {len(loaded)} file(s)…"):
        return get_finals(loaded, pipeline_params, keep_stages=keep_stages)


@st.cache_data(show_spinner=False, max_entries=16)
def _make_progress_echarts_cached(
    file_hash: str,
    pipeline_params: dict,
    _ds,
    title: str,
    x_unit: str,
    laser_nm: float | None,
    src_unit: str,
    native_type: str,
    x_range: tuple[float, float],
) -> dict:
    return make_progress_echarts(
        stage_dict(_ds),
        title=title,
        x_unit=x_unit,
        laser_nm=laser_nm,
        src_unit=src_unit,
        native_type=native_type,
        x_range=x_range,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _make_comparison_echarts_cached(
    file_hashes: str,
    pipeline_params: dict,
    _finals: dict,
    title: str,
    x_unit: str,
    laser_nm: float | None,
    src_unit: str,
    native_type: str,
    x_range: tuple[float, float] | None = None,
) -> dict:
    return make_comparison_echarts(
        _finals,
        title=title,
        x_unit=x_unit,
        laser_nm=laser_nm,
        src_unit=src_unit,
        native_type=native_type,
        x_range=x_range,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _make_final_echarts_cached(
    file_hash: str,
    pipeline_params: dict,
    _da,
    title: str,
    color_by: str,
    x_unit: str,
    laser_nm: float | None,
    src_unit: str,
    native_type: str,
) -> dict:
    return make_final_echarts(
        _da,
        title=title,
        color_by=color_by,
        x_unit=x_unit,
        laser_nm=laser_nm,
        src_unit=src_unit,
        native_type=native_type,
    )


def _render_progress_tab(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
    *,
    render_chart: bool = True,
) -> None:
    default_unit = UNIT_DEFAULT.get(ref_ds.spectral_units, "wavelength")
    current_unit = st.session_state.get("prog_x_unit", default_unit)

    x_native = ref_ds.da.coords[ref_ds.spectral_dim].values
    x_disp = convert_x(
        x_native,
        ref_ds.spectral_dim,
        current_unit,
        ref_ds.laser_nm,
        src_unit=ref_ds.spectral_unit,
        native_type=ref_ds.spectral_units,
    )
    disp_min = float(x_disp.min())
    disp_max = float(x_disp.max())

    col_unit, col_from, col_to, col_laser, col_title = st.columns([2, 1, 1, 1, 2])

    with col_unit:
        x_unit = st.selectbox(
            "Spectral units",
            X_UNIT_OPTIONS,
            format_func=X_UNIT_FMT.get,
            index=X_UNIT_OPTIONS.index(current_unit),
            key="prog_x_unit",
        )

    with col_from:
        _x_step = 0.01 if x_unit == "energy" else 1.0
        _x_fmt = "%.2f" if x_unit == "energy" else "%.0f"
        x_from = st.number_input(
            "From",
            value=min(disp_min, disp_max),
            key=f"prog_from_{x_unit}",
            format=_x_fmt,
            step=_x_step,
        )

    with col_to:
        x_to = st.number_input(
            "To",
            value=max(disp_min, disp_max),
            key=f"prog_to_{x_unit}",
            format=_x_fmt,
            step=_x_step,
        )

    laser = ref_ds.laser_nm
    with col_laser:
        if x_unit == "raman_shift" and laser is None:
            laser = st.number_input(
                "Laser (nm)",
                value=532.0,
                min_value=1.0,
                step=0.1,
                key="prog_laser_nm",
                help="Not found in file — enter the excitation wavelength.",
            )

    x_range = (min(x_from, x_to), max(x_from, x_to))

    default_title = next(iter(all_datasets)) if not multi else "Final spectra — all files"
    with col_title:
        chart_title = st.text_input("Chart title", value=default_title, key="prog_title")

    if not render_chart:
        return

    if multi:
        finals = {name: final_da(ds) for name, ds in all_datasets.items()}
        file_hashes = ",".join(sorted(loaded[name]["hash"] for name in all_datasets))
        opts = _make_comparison_echarts_cached(
            file_hashes,
            pipeline_params,
            finals,
            chart_title,
            x_unit,
            laser,
            ref_ds.spectral_unit,
            ref_ds.spectral_units,
            x_range=x_range,
        )
    else:
        name = next(iter(all_datasets))
        opts = _make_progress_echarts_cached(
            loaded[name]["hash"],
            pipeline_params,
            all_datasets[name],
            chart_title,
            x_unit,
            laser,
            ref_ds.spectral_unit,
            ref_ds.spectral_units,
            x_range,
        )

    st_echarts(opts, height="72vh", key="progress_chart")


_BROWSE_ALL = "All spectra"
_BROWSE_ONE = "Single spectrum"
_BROWSE_ROW = "Row"
_BROWSE_COL = "Column"
_BROWSE_MODES_MAP = [_BROWSE_ALL, _BROWSE_ONE, _BROWSE_ROW, _BROWSE_COL]
_BROWSE_MODES_LINE = [_BROWSE_ALL, _BROWSE_ONE]


def _clamp_index_state(key: str, n: int) -> None:
    """Keep a stepper's stored index inside the display range [1, n].

    The stepper state is stored in *display* numbering (the number_input is
    bound straight to it by key, so what is stored is what is shown); callers
    get the 0-based index back from :func:`_index_stepper`.

    Clamping is needed because the index widgets outlive the file they were set
    for: a switch to a smaller map (or a different scan geometry) would
    otherwise hand st.number_input a value above its own max and raise.
    """
    if key not in st.session_state:
        return
    try:
        st.session_state[key] = int(
            np.clip(
                int(st.session_state[key]), DISPLAY_BASE, max(n - 1 + DISPLAY_BASE, DISPLAY_BASE)
            )
        )
    except (TypeError, ValueError):
        del st.session_state[key]


def _step_index(key: str, delta: int, n: int) -> None:
    st.session_state[key] = int(
        np.clip(
            int(st.session_state.get(key, DISPLAY_BASE)) + delta,
            DISPLAY_BASE,
            max(n - 1 + DISPLAY_BASE, DISPLAY_BASE),
        )
    )


def _index_stepper(cols, label: str, key: str, n: int) -> int:
    """◀ / value / ▶ index picker rendered into three pre-made columns.

    Shows 1-based numbers (:data:`DISPLAY_BASE`) and returns the 0-based index,
    so it stays interchangeable with the Exclude Spectra tab's typed fields.

    The columns are passed in rather than created here: this widget already
    lives inside the page's right column, and Streamlit allows only one level
    of column nesting in the main area.
    """
    _clamp_index_state(key, n)
    lo, hi = DISPLAY_BASE, max(n - 1 + DISPLAY_BASE, DISPLAY_BASE)
    cur = int(st.session_state.get(key, lo))
    c_prev, c_val, c_next = cols
    c_prev.button(
        "◀",
        key=f"{key}_prev",
        width="stretch",
        disabled=cur <= lo,
        on_click=_step_index,
        args=(key, -1, n),
    )
    val = c_val.number_input(
        label,
        min_value=lo,
        max_value=hi,
        step=1,
        key=key,
    )
    c_next.button(
        "▶",
        key=f"{key}_next",
        width="stretch",
        disabled=cur >= hi,
        on_click=_step_index,
        args=(key, +1, n),
    )
    return int(val) - DISPLAY_BASE


def _render_browse_controls(da) -> tuple[Any, str, str]:
    """Browse picker for the Final tab: one spectrum, one row or one column.

    Returns ``(subset, mode, label)`` — ``subset`` is a view of ``da`` (1-D for
    a single spectrum, 2-D for a row/column), ``label`` names the selection for
    the chart title and caption. In ``All spectra`` mode the array is returned
    untouched and ``label`` is empty.
    """
    spectral_dim = da.dims[-1]
    spatial = [d for d in da.dims if d != spectral_dim]
    if not spatial:
        return da, _BROWSE_ALL, ""

    modes = _BROWSE_MODES_MAP if len(spatial) == 2 else _BROWSE_MODES_LINE
    if st.session_state.get("final_browse_mode") not in modes:
        st.session_state.pop("final_browse_mode", None)
    mode = (
        st.segmented_control(
            "Show",
            modes,
            key="final_browse_mode",
            default=_BROWSE_ALL,
            help=(
                "Step through the map one spectrum at a time, or plot a single "
                "map row / column. Indices are the same ones the Exclude Spectra "
                "tab uses."
            ),
        )
        or _BROWSE_ALL
    )

    if mode == _BROWSE_ALL:
        return da, mode, ""

    if len(spatial) == 1:
        n = int(da.sizes[spatial[0]])
        cols = st.columns([1, 3, 1, 8], vertical_alignment="bottom")
        i = _index_stepper(cols[:3], "Spectrum", "final_browse_i", n)
        return da.isel({spatial[0]: i}), mode, f"spectrum {to_display(i)}"

    n_row, n_col = int(da.sizes[spatial[0]]), int(da.sizes[spatial[1]])

    if mode == _BROWSE_ONE:
        cols = st.columns([1, 3, 1, 1, 3, 1, 2], vertical_alignment="bottom")
        r = _index_stepper(cols[:3], "Row", "final_browse_row", n_row)
        c = _index_stepper(cols[3:6], "Column", "final_browse_col", n_col)
        subset = da.isel({spatial[0]: r, spatial[1]: c})
        return (
            subset,
            mode,
            (f"pixel ({to_display(r)}, {to_display(c)}) · spectrum {to_display(r * n_col + c)}"),
        )

    if mode == _BROWSE_ROW:
        cols = st.columns([1, 3, 1, 8], vertical_alignment="bottom")
        r = _index_stepper(cols[:3], "Row", "final_browse_row", n_row)
        return da.isel({spatial[0]: r}), mode, f"row {to_display(r)} ({n_col} spectra)"

    cols = st.columns([1, 3, 1, 8], vertical_alignment="bottom")
    c = _index_stepper(cols[:3], "Column", "final_browse_col", n_col)
    return da.isel({spatial[1]: c}), mode, f"column {to_display(c)} ({n_row} spectra)"


def _render_final_tab(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
    *,
    render_chart: bool = True,
) -> None:
    if multi:
        view_mode = st.radio(
            "View",
            ["Comparison (all files)", "Single file"],
            horizontal=True,
            key="final_view_mode",
        )
    else:
        view_mode = "Single file"

    if view_mode == "Comparison (all files)":
        chart_title = st.text_input(
            "Chart title", value="Comparison — all files", key="fin_cmp_title"
        )
        x_unit, laser = render_axis_controls(
            "fin_cmp",
            ref_ds.laser_nm,
            native_type=ref_ds.spectral_units,
        )
        if render_chart:
            finals = {name: final_da(ds) for name, ds in all_datasets.items()}
            file_hashes = ",".join(sorted(loaded[name]["hash"] for name in all_datasets))
            st_echarts(
                _make_comparison_echarts_cached(
                    file_hashes,
                    pipeline_params,
                    finals,
                    chart_title,
                    x_unit,
                    laser,
                    ref_ds.spectral_unit,
                    ref_ds.spectral_units,
                ),
                height="72vh",
                key="final_comparison",
            )

    else:
        if multi:
            selected = st.selectbox(
                "Select file", list(all_datasets.keys()), key="final_file_select"
            )
        else:
            selected = next(iter(all_datasets))

        sel_ds: SpectralDataset = loaded[selected]["dataset"]
        da_full = final_da(all_datasets[selected])

        ctl1, ctl2 = st.columns([2, 2])
        color_by = ctl1.selectbox(
            "Color mode",
            ["index", "mean_dev"],
            format_func=lambda x: {
                "index": "Index (spectrum order)",
                "mean_dev": "Mean deviation",
            }[x],
            key="final_color_by",
        )
        chart_title = ctl2.text_input("Chart title", value=selected, key="fin_single_title")

        da_sel, browse_mode, browse_label = _render_browse_controls(da_full)

        n_spectra = int(da_sel.size // da_sel.shape[-1]) if da_sel.ndim > 1 else 1
        if browse_mode == _BROWSE_ALL and n_spectra > 5000:
            st.warning(
                f"Large dataset ({n_spectra} spectra). "
                "Only a subset of spectra is drawn in index mode "
                "(display is also downsampled to ~1,200 points/spectrum along "
                "the spectral axis). Exports and analysis stay full-resolution. "
                "Use **Show** above to step through spectra individually.",
                icon="⚠️",
            )

        x_unit, laser = render_axis_controls(
            "fin_single",
            sel_ds.laser_nm,
            native_type=sel_ds.spectral_units,
        )

        # Every spectrum of the selection is all-NaN — Clean Data dropped it or
        # the user excluded it. NaN is not valid JSON, so there is no chart to
        # draw; say which it is instead of rendering an empty axis.
        n_valid = int(
            np.sum(~np.all(np.isnan(da_sel.values.reshape(-1, da_sel.shape[-1])), axis=1))
        )
        if n_valid == 0:
            st.info(
                f"Nothing to plot — {browse_label or 'this file'} is entirely "
                "removed (Clean Data) or manually excluded."
            )
            return

        if not render_chart:
            return

        if browse_mode == _BROWSE_ALL:
            opts = _make_final_echarts_cached(
                loaded[selected]["hash"],
                pipeline_params,
                da_sel,
                chart_title,
                color_by,
                x_unit,
                laser,
                sel_ds.spectral_unit,
                sel_ds.spectral_units,
            )
        else:
            # Subsets are one row/column/pixel — cheap enough to build eagerly,
            # and the cache key would need the browse indices folded in.
            opts = make_final_echarts(
                da_sel,
                title=f"{chart_title} — {browse_label}",
                color_by=color_by,
                x_unit=x_unit,
                laser_nm=laser,
                src_unit=sel_ds.spectral_unit,
                native_type=sel_ds.spectral_units,
            )
            skipped = n_spectra - n_valid
            st.caption(
                f"Showing {browse_label}"
                + (f" · {skipped} removed/excluded spectra not drawn" if skipped else "")
            )

        st_echarts(opts, height="72vh", key="final_single")


def _pre_exclusion_da(ds) -> Any:
    """The final stage *before* manual exclusion, or the final stage itself.

    The selection map needs the un-excluded data underneath: showing the
    masked result would punch holes exactly where the user has to click to
    restore a spectrum.
    """
    if ds.attrs.get("final_var") != "excluded":
        return final_da(ds)
    stored = [v for v in ds.attrs["stage_vars"] if v in ds.data_vars and v != "excluded"]
    return ds[stored[-1]] if stored else final_da(ds)


@st.cache_data(show_spinner=False, max_entries=8)
def _selection_z_cached(file_hash: str, base_params: dict, _da) -> np.ndarray:
    """Integrated intensity behind the selection map.

    Keyed on the params *without* the exclusion mask — the underlying array is
    the pre-exclusion one, so it must not be recomputed on every mask edit.
    """
    return _da.sum(_da.dims[-1], min_count=1).values


def _selected_flat_indices(points: list[dict], n_col: int) -> list[int]:
    """Flat C-order pixel indices from a Plotly selection payload.

    The scatter layer is curve 1 and covers every pixel in row-major order, so
    ``point_index`` already is the flat index; ``customdata`` carries (row,
    column) as the authoritative cross-check — in *display* numbering, since it
    doubles as the hover label, hence the ``- DISPLAY_BASE``.
    """
    out: list[int] = []
    for p in points:
        if p.get("curve_number") not in (None, 1):
            continue
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) >= 2:
            r = int(cd[0]) - DISPLAY_BASE
            c = int(cd[1]) - DISPLAY_BASE
            out.append(r * n_col + c)
            continue
        i = p.get("point_index", p.get("point_number"))
        if i is not None:
            out.append(int(i))
    return out


def _render_selection_tab(
    all_datasets: dict,
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    *,
    render_chart: bool = True,
) -> None:
    """Interactive pixel picker — click / box / lasso to exclude or restore."""
    fname = st.session_state.get("excl_file") or next(iter(all_datasets), None)
    if fname not in all_datasets:
        st.info("Select a file in the Exclude Spectra tab.")
        return

    ds_meta: SpectralDataset = loaded[fname]["dataset"]
    if not ds_meta.is_map:
        st.info(
            "Interactive selection needs a raster map. For line scans and "
            "series, use the index fields in the Exclude Spectra tab."
        )
        return

    da_ctx = _pre_exclusion_da(all_datasets[fname])
    spectral_dim = ds_meta.spectral_dim
    shape = spatial_shape(da_ctx, spectral_dim)
    n_row, n_col = shape

    mode = st.session_state.get("excl_mode") or "Exclude"
    exclude = mode != "Restore"
    st.caption(
        f"Mode: **{mode}** — drag a box, lasso a region, or click one pixel. "
        "Change the mode in the Exclude Spectra tab. Double-click clears the selection."
    )
    if n_row * n_col > 100_000:
        st.caption(
            f"⚠ {n_row * n_col:,} pixels — the picker may feel sluggish; "
            "the index fields are faster for bulk edits."
        )

    base_params = {k: v for k, v in pipeline_params.items() if k != "excl"}
    z = _selection_z_cached(loaded[fname]["hash"], base_params, da_ctx)

    user_mask = get_mask(fname, shape)
    auto_mask = _cd_removed_mask(da_ctx, spectral_dim)

    colorscale, map_opacity = render_map_display_controls("excl")
    if not render_chart:
        st.caption(
            f"{int(user_mask.sum())} excluded · {int((auto_mask & ~user_mask).sum())} "
            f"auto-removed · {n_row * n_col} total"
        )
        return

    fig = make_selection_map_fig(
        z,
        da_ctx.coords[da_ctx.dims[0]].values,
        da_ctx.coords[da_ctx.dims[1]].values,
        ds_meta.image_arr,
        ds_meta.image_meta,
        user_mask,
        auto_mask,
        colorscale=colorscale,
        title=fname,
        map_opacity=map_opacity,
    )
    event = st.plotly_chart(
        fig,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key="excl_map",
        width="stretch",
        height=600,
        config=PLOTLY_CONFIG,
    )

    points = ((event or {}).get("selection") or {}).get("points") or []
    idx = _selected_flat_indices(points, n_col)
    # Streamlit keeps the selection in session state across reruns, so the
    # same payload arrives again on every subsequent rerun — apply each
    # distinct (file, mode, selection) exactly once.
    signature = (fname, mode, tuple(idx))
    if idx and st.session_state.get("_excl_last_selection") != signature:
        st.session_state["_excl_last_selection"] = signature
        set_mask(fname, apply_selection(user_mask, flat=idx, exclude=exclude))
        # scope="app": this fragment owns the chart, but the left column
        # rebuilds sl_pipeline_params and the other tabs read the new mask.
        st.rerun(scope="app")

    st.caption(
        f"{int(user_mask.sum())} excluded · {int((auto_mask & ~user_mask).sum())} "
        f"auto-removed · {n_row * n_col} total"
    )


def _chart_tab_steps_active(tab_prog, tab_final, tab_sel) -> bool:
    """True when the Steps chart tab is selected (including its default on first load)."""
    if tab_final.open is True or tab_sel.open is True:
        return False
    return tab_prog.open is not False


def _render_charts_section(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any],
    multi: bool,
    ref_ds: SpectralDataset,
) -> tuple[dict, list[str]]:
    """Right-column chart tabs; returns ``(all_datasets, errors)``."""
    tab_prog, tab_final, tab_sel = st.tabs(
        ["Steps", "Final", "Selection"],
        on_change="rerun",
    )
    keep_stages = _chart_tab_steps_active(tab_prog, tab_final, tab_sel)
    all_datasets, errors = _run_preprocessing(
        loaded,
        pipeline_params,
        keep_stages=keep_stages,
    )

    with tab_prog:
        _render_progress_tab(
            all_datasets,
            loaded,
            pipeline_params,
            multi,
            ref_ds,
            render_chart=tab_prog.open is True,
        )
    with tab_final:
        _render_final_tab(
            all_datasets,
            loaded,
            pipeline_params,
            multi,
            ref_ds,
            render_chart=tab_final.open is True,
        )
    with tab_sel:
        _render_selection_tab(
            all_datasets,
            loaded,
            pipeline_params,
            render_chart=tab_sel.open is True,
        )

    return all_datasets, errors


# ────────────────────────────── Page assembly ──────────────────────────────


def render_preprocessing_page() -> None:
    """Preprocessing page: pipeline controls (left) + staged and final charts (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    ensure_pipeline_widget_defaults()
    _restore_widget_state()

    processing_ok: bool = st.session_state.get("sl_processing_ok", False)
    ref_ds: SpectralDataset = next(iter(loaded.values()))["dataset"]

    left, right = st.columns([1, 2], gap="medium")

    with left:
        pipeline_params, cd_visual_slot, excl_visual_slot = _render_preprocessing_params(
            processing_ok, loaded
        )
        st.session_state["sl_pipeline_params"] = pipeline_params

    with right:
        nothing_enabled = not any(
            pipeline_params.get(k)
            for k in ("norm1_enabled", "cd_enabled", "crr_enabled", "denoise_enabled")
        ) and not (pipeline_params.get("excl") or {}).get("masks")
        if nothing_enabled:
            st.info(
                "No processing enabled — the charts show raw data. "
                "Use a Quick Setup preset or enable steps on the left."
            )

        multi = len(loaded) > 1
        all_datasets, errors = _render_charts_section(
            loaded,
            pipeline_params,
            multi,
            ref_ds,
        )

        for err in errors:
            st.error(f"Processing error — {err}")
        if not all_datasets:
            st.stop()

    # Fill the Clean Data / Exclude tab placeholders now that the pipeline has
    # run.
    if cd_visual_slot is not None:
        _render_cd_removed_visual(cd_visual_slot, all_datasets, loaded)
    if excl_visual_slot is not None:
        _render_excl_visual(excl_visual_slot, all_datasets, loaded)

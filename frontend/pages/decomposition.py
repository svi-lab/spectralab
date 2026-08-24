"""Decomposition page: NMF or MCR-ALS pattern discovery across a map scan."""

from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
from backend.spectra_decomposer import Decomposer, compute_nmf_diagnostic_curve
from backend.spectra_mcr import MCRDecomposer, compute_mcr_rank_svd

from ..charts import (
    make_components_echarts,
    make_mcr_ambiguity_echarts,
    make_mcr_scree_echarts,
    make_nmf_diagnostic_echarts,
)
from ..controls import (
    render_map_display_controls,
    render_mcr_params,
    render_nmf_params,
)
from ..export_utils import mcr_to_npz, nmf_to_npz
from ..map_chart import make_scalar_map_fig
from ..pipeline_cache import default_pipeline_params, final_da, get_finals

# No spectral-unit selector on this page: every component / ambiguity chart is
# drawn on an energy scale, same deliberate exception as the Map Analysis page.
# Safe without a laser wavelength here because the page already refuses
# anything but PL (Nanometer / ElectronVolt natives) further down.
_X_UNIT = "energy"


def _resolve_reference_spectrum(all_datasets, loaded, ref_file, target_x):
    """Mean spectrum of ``ref_file`` (processed if available), resampled onto
    ``target_x`` and shifted to a baseline of 0 — ready to pin an MCR
    component to. Both files are PL, so their axes share units."""
    da_ref = final_da(all_datasets.get(ref_file))
    if da_ref is None:
        da_ref = loaded[ref_file]["dataset"].da
    sdim = da_ref.dims[-1]
    non_spectral = tuple(range(da_ref.ndim - 1))
    ref_y = np.nanmean(da_ref.values, axis=non_spectral) if non_spectral else da_ref.values
    ref_x = np.asarray(da_ref.coords[sdim].values, dtype=float)
    order = np.argsort(ref_x)
    ref_y_resampled = np.interp(
        np.asarray(target_x, dtype=float), ref_x[order], np.asarray(ref_y, dtype=float)[order]
    )
    ref_y_resampled = ref_y_resampled - np.nanmin(ref_y_resampled)
    return np.clip(ref_y_resampled, 0, None)


def render_decomposition_page() -> None:
    """Decomposition page: method + parameters (left), results (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    pipeline_params = st.session_state.get("sl_pipeline_params") or default_pipeline_params()
    with st.spinner("Preparing data…"):
        all_datasets, _errors = get_finals(loaded, pipeline_params)

    map_candidates = {name: entry for name, entry in loaded.items() if entry["dataset"].is_map}
    if not map_candidates:
        st.info(
            "Decomposition requires a map scan (3D row/column/spectral data) "
            "to produce spatial abundance/concentration maps."
        )
        return

    left, right = st.columns([1, 2], gap="medium")

    with left:
        if len(map_candidates) > 1:
            map_name = st.selectbox(
                "Select file", list(map_candidates.keys()), key="nmf_file_select"
            )
        else:
            map_name = next(iter(map_candidates))
        ds: SpectralDataset = map_candidates[map_name]["dataset"]

    if ds.measurement_kind != "PL":
        msg = "Decomposition is available for PL data only (Nanometer/ElectronVolt axes)."
        with left:
            st.info(msg)
        with right:
            st.info(msg)
        return

    da_map = final_da(all_datasets.get(map_name))
    if da_map is None:
        with right:
            st.warning(
                "Processing result not available for this file. Visit the Preprocessing page first."
            )
        return

    with left:
        with st.container(border=True):
            st.markdown('<p class="section-header">Method</p>', unsafe_allow_html=True)
            method = st.radio(
                "Decomposition method",
                ["NMF", "MCR-ALS"],
                index=0,
                key="decomp_method",
                help=(
                    "**NMF** (default): a fast statistical basis for recurring "
                    "spectral shapes — useful, but its components are not tied "
                    "to a physical model as directly.\n\n"
                    "**MCR-ALS**: resolves physically interpretable "
                    "pure-component spectra + concentration maps under "
                    "non-negativity, and reports how uniquely each component "
                    "is resolved."
                ),
            )

    if method == "MCR-ALS":
        _render_mcr(left, right, loaded, all_datasets, da_map, ds, map_name)
    else:
        _render_nmf(left, right, da_map, ds, map_name)


# --------------------------------------------------------------------------- #
# MCR-ALS
# --------------------------------------------------------------------------- #
def _render_mcr(left, right, loaded, all_datasets, da_map, ds, map_name):
    spectral_dim = da_map.dims[-1]
    target_x = da_map.coords[spectral_dim].values

    with left:
        with st.container(border=True):
            st.markdown(
                '<p class="section-header">Number of Components</p>', unsafe_allow_html=True
            )
            st.caption(
                "Use the SVD scree to see how many spectra rise above the "
                "noise floor — pick the count just before the bars go flat."
            )
            run_rank = st.button("Estimate rank (SVD)", key="mcr_run_rank")
            n_components = st.number_input(
                "Number of components",
                value=2,
                min_value=1,
                step=1,
                key="mcr_n_components",
                help="How many distinct emission species to resolve. Read it off the SVD scree.",
            )

        with st.container(border=True):
            st.markdown('<p class="section-header">Constraints & Run</p>', unsafe_allow_html=True)
            st.caption("Non-negativity is always applied to both spectra and concentrations.")
            use_eq = st.checkbox(
                "Fix one component to a reference spectrum",
                key="mcr_use_equality",
                help=(
                    "Pin one component's spectrum to a measured reference — "
                    "e.g. a bare-substrate PL spectrum — anchoring the result "
                    "physically and cutting rotational ambiguity."
                ),
            )
            equality_spectrum = None
            equality_index = 0
            if use_eq:
                other_files = [n for n in loaded if n != map_name]
                if not other_files:
                    st.warning("Load a reference file (e.g. a bare substrate) to use this.")
                else:
                    ss_struct = st.session_state.get("sl_sample_structure", {})
                    sub_files = [
                        n
                        for n in other_files
                        if ss_struct.get(n, {}).get("sample_type") == "substrate"
                    ]
                    candidates = sub_files + [n for n in other_files if n not in sub_files]
                    ref_file = st.selectbox(
                        "Reference file",
                        candidates,
                        key="mcr_ref_file",
                        format_func=lambda n: f"{n} — substrate" if n in sub_files else n,
                    )
                    equality_index = st.selectbox(
                        "Apply to component",
                        range(int(n_components)),
                        format_func=lambda i: f"Component {i + 1}",
                        key="mcr_equality_index",
                    )
                    equality_spectrum = _resolve_reference_spectrum(
                        all_datasets, loaded, ref_file, target_x
                    )

            mcr_params = render_mcr_params()
            quant_amb = st.checkbox(
                "Quantify rotational ambiguity",
                value=True,
                key="mcr_quant_amb",
                help=(
                    "Runs the feasible-band f_max − f_min analysis after the "
                    "fit — how uniquely each component is resolved. Adds a few "
                    "seconds; turn off for very large maps."
                ),
            )
            run_decompose = st.button("Run MCR-ALS", key="mcr_run_decompose", type="primary")

    with right:
        if run_rank:
            with st.spinner("Computing SVD scree…"):
                try:
                    st.session_state["sl_mcr_rank"] = compute_mcr_rank_svd(
                        da_map.values,
                        random_state=mcr_params["random_state"],
                    )
                except ValueError as exc:
                    st.error(f"Could not compute SVD scree: {exc}")

        if run_decompose:
            with st.spinner("Running MCR-ALS (alternating NNLS)…"):
                try:
                    decomposer = MCRDecomposer(
                        n_components=int(n_components),
                        equality_spectrum=equality_spectrum,
                        equality_index=int(equality_index),
                        quantify_ambiguity=bool(quant_amb),
                        **mcr_params,
                    )
                    _, payload = decomposer.decompose(da_map)
                    st.session_state["sl_mcr_result"] = {
                        "components": payload["components"],
                        "abundances": payload["abundances"],
                        "meta": payload["meta"],
                        "ambiguity": payload.get("ambiguity"),
                        "spectral_coords": target_x,
                        "spectral_dim": spectral_dim,
                        "file_name": map_name,
                        "method": "mcr-als",
                    }
                except ValueError as exc:
                    st.error(f"MCR-ALS failed: {exc}")

        mcr_result = st.session_state.get("sl_mcr_result")
        if mcr_result and mcr_result["file_name"] == map_name:
            _render_mcr_results(mcr_result, ds, map_name)
        elif mcr_result is not None:
            st.caption(
                f"Last MCR-ALS result was for **{mcr_result['file_name']}** — run again for **{map_name}**."
            )

        rank = st.session_state.get("sl_mcr_rank")
        if rank:
            st.divider()
            rank_title = st.text_input("Chart title", value="SVD Scree", key="mcr_rank_title")
            st_echarts(make_mcr_scree_echarts(rank, title=rank_title), height="72vh")
            if rank["subsampled"]:
                st.caption(
                    f"Scree computed on a random subsample of {rank['n_pixels_used']} "
                    f"of {rank['n_pixels_total']} pixels for speed. The final "
                    "decomposition always uses every pixel."
                )


def _render_mcr_results(mcr_result, ds, map_name):
    st.caption(
        "**Physical read:** each component spectrum is a resolved pure-emission "
        "profile and each map is its relative concentration. How trustworthy "
        "those spectra are depends on the rotational ambiguity — see the "
        "**Ambiguity** tab, not the fit quality alone."
    )

    tab_components, tab_maps, tab_amb, tab_stats = st.tabs(
        ["Component Spectra", "Concentration Maps", "Ambiguity", "Statistics"]
    )

    with tab_components:
        comp_title = st.text_input(
            "Chart title", value="Pure-Component Spectra", key="mcr_comp_title"
        )
        st_echarts(
            make_components_echarts(
                mcr_result["components"],
                mcr_result["spectral_coords"],
                mcr_result["spectral_dim"],
                title=comp_title,
                x_unit=_X_UNIT,
                laser_nm=ds.laser_nm,
                src_unit=ds.spectral_unit,
                native_type=ds.spectral_units,
            ),
            height="72vh",
        )

    with tab_maps:
        n_comp = mcr_result["components"].shape[0]
        c_comp, c_display = st.columns([1, 2])
        comp_idx = c_comp.selectbox(
            "Component",
            range(n_comp),
            format_func=lambda i: f"Component {i + 1}",
            key="mcr_map_component_select",
        )
        with c_display:
            colorscale, map_opacity = render_map_display_controls("mcr_map", inline=True)
        abundances = mcr_result["abundances"]
        z = abundances.isel(component=comp_idx).values
        row_coords = abundances.coords["row"].values
        col_coords = abundances.coords["column"].values
        fig = make_scalar_map_fig(
            z,
            row_coords,
            col_coords,
            ds.image_arr,
            ds.image_meta,
            cbar_label=f"Component {comp_idx + 1} concentration",
            title=f"{map_name} — Component {comp_idx + 1}",
            colorscale=colorscale,
            map_opacity=map_opacity,
        )
        st.plotly_chart(fig, width="stretch", height=550)

    with tab_amb:
        _render_ambiguity_tab(mcr_result, ds)

    with tab_stats:
        meta = mcr_result["meta"]
        st.markdown(
            f'<div class="info-box">'
            f"<b>n_components:</b> {meta['n_components']}<br>"
            f"<b>n_iter:</b> {meta['n_iter']} / max_iter={meta['max_iter']}<br>"
            f"<b>Converged:</b> {meta['converged']}<br>"
            f"<b>Lack of fit (%LOF):</b> {meta['lof']:.4g}<br>"
            f"<b>Variance explained:</b> {meta['fraction_var_explained']:.4f}<br>"
            f"<b>Constraints:</b> {', '.join(meta['constraints'])}<br>"
            f"<b>n_spectra:</b> {meta['n_spectra']}, "
            f"<b>n_spectral:</b> {meta['n_spectral']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        if not meta["converged"]:
            st.warning(
                "This fit did not reach the convergence threshold within "
                "max_iter — results may be less stable. Consider raising "
                "max_iter or loosening the threshold in Advanced MCR-ALS parameters."
            )
        st.download_button(
            "Export MCR result (.npz)",
            mcr_to_npz(mcr_result),
            file_name=f"{map_name}_mcr.npz",
            key="mcr_export_npz",
        )


def _render_ambiguity_tab(mcr_result, ds):
    amb = mcr_result.get("ambiguity")
    if not amb:
        st.info(
            "Ambiguity was not quantified for this run. Re-run with "
            "**Quantify rotational ambiguity** enabled to see how uniquely "
            "each component is resolved."
        )
        return
    if not amb.get("ok"):
        st.warning(
            "The feasible-band optimiser could not resolve the ambiguity bands "
            f"({amb.get('reason', 'no feasible solution found')}). This often "
            "means the components are strongly overlapping — treat the resolved "
            "spectra with caution."
        )
        return

    amb_title = st.text_input("Chart title", value="Rotational Ambiguity", key="mcr_amb_title")
    st_echarts(make_mcr_ambiguity_echarts(amb, title=amb_title), height="60vh")

    st.caption(
        "**How to read this:** each bar is the range of relative signal a "
        "component could still contribute across all equally-good (non-negative) "
        "solutions. Near zero → the component is essentially uniquely resolved. "
        "A wide bar → several factorizations fit equally well, so that "
        "component's exact shape/amplitude is not proven."
    )

    f_range = amb["f_range"]
    source = amb.get("dominant_source", [""] * len(f_range))
    for i, (r, src) in enumerate(zip(f_range, source)):
        if r != r:  # NaN
            verdict = "could not be bounded"
        elif r < 0.05:
            verdict = f"uniquely resolved (band {r:.3f})"
        elif r < 0.2:
            verdict = f"moderate ambiguity (band {r:.3f})"
        else:
            verdict = f"**high ambiguity** (band {r:.3f})"
        where = f" — mostly in the {src}" if src in ("spectrum", "concentration", "both") else ""
        st.caption(f"Component {i + 1}: {verdict}{where}.")

    boundary = amb.get("boundary")
    if boundary is not None:
        st.markdown(
            f"**Boundary spectra — Component {boundary['component'] + 1}** "
            "(the two extreme feasible shapes for the most ambiguous component):"
        )
        st_echarts(
            make_components_echarts(
                np.vstack([boundary["s_min"], boundary["s_max"]]),
                mcr_result["spectral_coords"],
                mcr_result["spectral_dim"],
                title="Feasible-band boundary spectra",
                x_unit=_X_UNIT,
                laser_nm=ds.laser_nm,
                src_unit=ds.spectral_unit,
                native_type=ds.spectral_units,
            ),
            height="60vh",
        )


# --------------------------------------------------------------------------- #
# NMF (unchanged behaviour)
# --------------------------------------------------------------------------- #
def _render_nmf(left, right, da_map, ds, map_name):
    with left:
        with st.container(border=True):
            st.markdown('<p class="section-header">Diagnostic Curve</p>', unsafe_allow_html=True)
            k_max = st.number_input(
                "Max components to test",
                value=8,
                min_value=2,
                max_value=20,
                step=1,
                key="nmf_kmax",
                help=(
                    "Fits NMF for 1..this many components and plots how "
                    "reconstruction error / variance-explained trade off, so "
                    "you can pick the count from where the curve elbows."
                ),
            )
            run_diag = st.button("Compute diagnostic curve", key="nmf_run_diagnostic")

        with st.container(border=True):
            st.markdown('<p class="section-header">Decomposition</p>', unsafe_allow_html=True)
            n_components = st.number_input(
                "n_components (k)",
                value=3,
                min_value=1,
                step=1,
                key="nmf_n_components",
                help="Pick this using the diagnostic curve above — there is no automatic/hidden k selection.",
            )
            nmf_params = render_nmf_params()
            run_decompose = st.button(
                "Run NMF decomposition", key="nmf_run_decompose", type="primary"
            )

    with right:
        if run_diag:
            with st.spinner("Sweeping NMF components…"):
                try:
                    diag = compute_nmf_diagnostic_curve(
                        da_map.values,
                        k_max=int(k_max),
                        init=nmf_params["init"],
                        random_state=nmf_params["random_state"],
                    )
                    st.session_state["sl_nmf_diagnostic"] = diag
                except ValueError as exc:
                    st.error(f"Could not compute diagnostic curve: {exc}")

        if run_decompose:
            with st.spinner("Running NMF…"):
                try:
                    decomposer = Decomposer(n_components=int(n_components), **nmf_params)
                    _, payload = decomposer.decompose(da_map)
                    spectral_dim = da_map.dims[-1]
                    st.session_state["sl_nmf_result"] = {
                        "components": payload["components"],
                        "abundances": payload["abundances"],
                        "meta": payload["meta"],
                        "spectral_coords": da_map.coords[spectral_dim].values,
                        "spectral_dim": spectral_dim,
                        "file_name": map_name,
                    }
                except ValueError as exc:
                    st.error(f"NMF decomposition failed: {exc}")

        nmf_result = st.session_state.get("sl_nmf_result")
        if nmf_result and nmf_result["file_name"] == map_name:
            st.caption(
                "**Rotational-ambiguity caveat:** NMF components are not guaranteed "
                "to be unique physical end-members — different factorizations can "
                "reconstruct the data equally well. Treat components as a useful "
                "basis for recurring spectral shapes, not as proven distinct "
                "chemical species, unless corroborated by independent evidence."
            )

            tab_components, tab_maps, tab_stats = st.tabs(
                ["Component Spectra", "Abundance Maps", "Statistics"]
            )

            with tab_components:
                comp_title = st.text_input(
                    "Chart title", value="Component Spectra", key="nmf_comp_title"
                )
                st_echarts(
                    make_components_echarts(
                        nmf_result["components"],
                        nmf_result["spectral_coords"],
                        nmf_result["spectral_dim"],
                        title=comp_title,
                        x_unit=_X_UNIT,
                        laser_nm=ds.laser_nm,
                        src_unit=ds.spectral_unit,
                        native_type=ds.spectral_units,
                    ),
                    height="72vh",
                )

            with tab_maps:
                n_comp = nmf_result["components"].shape[0]
                c_comp, c_display = st.columns([1, 2])
                comp_idx = c_comp.selectbox(
                    "Component",
                    range(n_comp),
                    format_func=lambda i: f"Component {i + 1}",
                    key="nmf_map_component_select",
                )
                with c_display:
                    colorscale, map_opacity = render_map_display_controls("nmf_map", inline=True)
                abundances = nmf_result["abundances"]
                z = abundances.isel(component=comp_idx).values
                row_coords = abundances.coords["row"].values
                col_coords = abundances.coords["column"].values
                fig = make_scalar_map_fig(
                    z,
                    row_coords,
                    col_coords,
                    ds.image_arr,
                    ds.image_meta,
                    cbar_label=f"Component {comp_idx + 1} abundance",
                    title=f"{map_name} — Component {comp_idx + 1}",
                    colorscale=colorscale,
                    map_opacity=map_opacity,
                )
                st.plotly_chart(fig, width="stretch", height=550)

            with tab_stats:
                meta = nmf_result["meta"]
                st.markdown(
                    f'<div class="info-box">'
                    f"<b>n_components:</b> {meta['n_components']}<br>"
                    f"<b>init:</b> {meta['init']}<br>"
                    f"<b>n_iter:</b> {meta['n_iter']} / max_iter={meta['max_iter']}<br>"
                    f"<b>Converged:</b> {meta['converged']}<br>"
                    f"<b>Reconstruction error:</b> {meta['reconstruction_err']:.4g}<br>"
                    f"<b>Variance-explained proxy:</b> {meta['fraction_var_explained']:.4f}<br>"
                    f"<b>n_spectra:</b> {meta['n_spectra']}, "
                    f"<b>n_spectral:</b> {meta['n_spectral']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if not meta["converged"]:
                    st.warning(
                        "This fit did not fully converge within max_iter — results "
                        "may be less stable. Consider raising max_iter in Advanced "
                        "NMF parameters."
                    )
                st.download_button(
                    "Export NMF result (.npz)",
                    nmf_to_npz(nmf_result),
                    file_name=f"{map_name}_nmf.npz",
                    key="nmf_export_npz",
                )
        elif nmf_result is not None:
            st.caption(
                f"Last NMF result was for **{nmf_result['file_name']}** — run again for **{map_name}**."
            )

        diag = st.session_state.get("sl_nmf_diagnostic")
        if diag:
            st.divider()
            diag_title = st.text_input(
                "Chart title", value="NMF Diagnostic Curve", key="nmf_diag_title"
            )
            st_echarts(make_nmf_diagnostic_echarts(diag, title=diag_title), height="72vh")
            n_not_converged = sum(1 for c in diag["converged"] if not c)
            if n_not_converged:
                st.caption(
                    f"⚠ {n_not_converged} of {len(diag['k_values'])} component counts "
                    "did not fully converge within max_iter (hollow markers above) — "
                    "consider raising max_iter in Advanced NMF parameters."
                )
            if diag["subsampled"]:
                st.caption(
                    f"Diagnostic computed on a random subsample of {diag['n_pixels_used']} "
                    f"of {diag['n_pixels_total']} pixels for speed. The final decomposition "
                    "always uses every pixel."
                )

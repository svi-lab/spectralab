# -*- coding: utf-8 -*-
"""Decomposition page: NMF pattern discovery across a map scan."""

from __future__ import annotations

import streamlit as st
from streamlit_echarts import st_echarts

from backend._shared.dataset import SpectralDataset
from backend.spectra_decomposer import Decomposer, compute_nmf_diagnostic_curve

from ..charts import make_components_echarts, make_nmf_diagnostic_echarts
from ..controls import render_axis_controls, render_nmf_params
from ..map_chart import make_scalar_map_fig
from ..pipeline_cache import default_pipeline_params, get_finals


def render_decomposition_page() -> None:
    """Decomposition page: NMF parameters (left) + components/maps/stats (right)."""
    loaded = st.session_state.get("sl_loaded")
    if not loaded:
        st.info("Upload files in the sidebar to get started.")
        st.stop()

    pipeline_params = st.session_state.get("sl_pipeline_params") or default_pipeline_params()
    with st.spinner("Preparing data…"):
        _, all_finals, _errors = get_finals(loaded, pipeline_params)

    map_candidates = {
        name: entry for name, entry in loaded.items()
        if entry["dataset"].is_map
    }
    if not map_candidates:
        st.info(
            "NMF decomposition requires a map scan (3D row/column/spectral "
            "data) to produce spatial abundance maps."
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
        with left:
            st.info("NMF decomposition is available for PL data only (Nanometer/ElectronVolt axes).")
        with right:
            st.info("NMF decomposition is available for PL data only (Nanometer/ElectronVolt axes).")
        return

    da_map = all_finals.get(map_name)
    if da_map is None:
        with right:
            st.warning("Processing result not available for this file. Visit the Preprocessing page first.")
        return

    with left:
        st.markdown('<p class="section-header">Display</p>', unsafe_allow_html=True)
        x_unit, laser_nm = render_axis_controls(
            "nmf", ds.laser_nm, native_type=ds.spectral_units,
        )

        st.markdown('<p class="section-header">Diagnostic Curve</p>', unsafe_allow_html=True)
        k_max = st.number_input(
            "k_max to sweep", value=8, min_value=2, max_value=20, step=1,
            key="nmf_kmax",
            help=(
                "Sweeps n_components = 1..k_max and reports reconstruction "
                "error / variance-explained at each step, so you can pick k "
                "from where the curve elbows instead of an automatic choice."
            ),
        )
        run_diag = st.button("Compute diagnostic curve", key="nmf_run_diagnostic")

        st.markdown('<p class="section-header">Decomposition</p>', unsafe_allow_html=True)
        n_components = st.number_input(
            "n_components (k)", value=3, min_value=1, step=1,
            key="nmf_n_components",
            help="Pick this using the diagnostic curve above — there is no automatic/hidden k selection.",
        )
        nmf_params = render_nmf_params()
        run_decompose = st.button("Run NMF decomposition", key="nmf_run_decompose", type="primary")

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

        diag = st.session_state.get("sl_nmf_diagnostic")
        if diag:
            st_echarts(make_nmf_diagnostic_echarts(diag), height="320px")
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
                    "below always uses every pixel."
                )

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
                st_echarts(
                    make_components_echarts(
                        nmf_result["components"],
                        nmf_result["spectral_coords"],
                        nmf_result["spectral_dim"],
                        x_unit=x_unit, laser_nm=laser_nm,
                        src_unit=ds.spectral_unit, native_type=ds.spectral_units,
                    ),
                    height="450px",
                )

            with tab_maps:
                n_comp = nmf_result["components"].shape[0]
                comp_idx = st.selectbox(
                    "Component", range(n_comp),
                    format_func=lambda i: f"Component {i + 1}",
                    key="nmf_map_component_select",
                )
                abundances = nmf_result["abundances"]
                z = abundances.isel(component=comp_idx).values
                row_coords = abundances.coords["row"].values
                col_coords = abundances.coords["column"].values
                fig = make_scalar_map_fig(
                    z, row_coords, col_coords,
                    ds.image_arr, ds.image_meta,
                    cbar_label=f"Component {comp_idx + 1} abundance",
                    title=f"{map_name} — Component {comp_idx + 1}",
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
        elif nmf_result is not None:
            st.caption(f"Last NMF result was for **{nmf_result['file_name']}** — run again for **{map_name}**.")

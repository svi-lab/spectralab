"""Reusable Streamlit widgets for the WDF Viewer."""

from __future__ import annotations

from typing import Any

import streamlit as st

X_UNIT_OPTIONS = ["wavelength", "energy", "wavenumber", "raman_shift"]
X_UNIT_FMT = {
    "wavelength": "Wavelength (nm)",
    "energy": "Energy (eV)",
    "wavenumber": "Wavenumber (cm⁻¹)",
    "raman_shift": "Raman Shift (cm⁻¹)",
}
UNIT_DEFAULT = {
    "RamanShift": "raman_shift",
    "Wavenumber": "wavenumber",
    "Nanometer": "energy",
    "ElectronVolt": "energy",
}

# Engine option labels — shared with the Preprocessing page's Quick Setup
# presets (per-spectrum / collection), which seed the widget keys with these
# exact strings.
CRR_ENGINE_1D = "1D — per spectrum"
CRR_ENGINE_2D3D = "2D / 3D — collection & spatial"
DENOISE_ENGINE_PCA = "PCA — population-based"
DENOISE_ENGINE_SMOOTHER = "Smoother — per spectrum"


def render_axis_controls(
    key_prefix: str,
    laser_nm: float | None,
    native_type: str = "",
) -> tuple:
    """Render x-axis unit selector (and laser input if needed).

    Returns (x_unit, effective_laser).
    """
    col_u, col_laser = st.columns([1, 1])
    default_unit = UNIT_DEFAULT.get(native_type, "wavelength")
    x_unit = col_u.selectbox(
        "Spectral units",
        X_UNIT_OPTIONS,
        format_func=X_UNIT_FMT.get,
        index=X_UNIT_OPTIONS.index(default_unit),
        key=f"{key_prefix}_x_unit",
    )

    effective_laser = laser_nm
    if x_unit == "raman_shift" and laser_nm is None:
        effective_laser = col_laser.number_input(
            "Laser wavelength (nm)",
            value=532.0,
            min_value=1.0,
            step=0.1,
            key=f"{key_prefix}_laser_nm",
            help="Not found in file — enter the excitation wavelength.",
        )

    return x_unit, effective_laser


MAP_COLORSCALES = ["Viridis", "Plasma", "Inferno", "Hot", "RdBu_r", "Turbo"]


def render_map_display_controls(key_prefix: str, inline: bool = False) -> tuple[str, float]:
    """Render the shared map-display widgets: colorscale + map opacity.

    Returns (colorscale, map_opacity). The heatmap is drawn above the
    white-light photo, so opacity sets the map/photo blend. ``inline``
    puts the two widgets side by side instead of stacked.
    """
    col_scale, col_opacity = st.columns(2) if inline else (st.container(), st.container())
    colorscale = col_scale.selectbox(
        "Colorscale",
        MAP_COLORSCALES,
        key=f"{key_prefix}_colorscale",
    )
    map_opacity = col_opacity.slider(
        "Map opacity",
        min_value=0.2,
        max_value=1.0,
        value=0.75,
        step=0.05,
        key=f"{key_prefix}_opacity",
    )
    return colorscale, map_opacity


def render_clean_data_params() -> dict[str, Any]:
    """Render CleanData parameter widget. Returns cd_params dict."""
    n_zeros = st.number_input(
        "Consecutive zeros to flag",
        value=10,
        min_value=1,
        step=1,
        key="cd_n_zeros",
        help=(
            "A spectrum is flagged as saturated when it contains at least "
            "this many **consecutive** channels equal to exactly 0 — the "
            "signature of ADC clipping. Raise this threshold if short "
            "zero runs appear legitimately in your data (e.g. detector "
            "gaps). Lower it to catch partially saturated spectra."
        ),
    )
    return {"n_zeros": int(n_zeros)}


def render_crr_params() -> dict[str, Any]:
    """Render CosmicRayRemover parameter widgets. Returns crr_params dict."""
    engine = st.selectbox(
        "Engine mode",
        options=[CRR_ENGINE_1D, CRR_ENGINE_2D3D],
        key="crr_engine_mode",
        help=(
            "**1D — per spectrum:** each spectrum is corrected independently "
            "using a median filter + MAD threshold. Simple, fast, and "
            "predictable for any data shape.\n\n"
            "**2D / 3D — collection & spatial:** uses the full population as "
            "a reference — global median or PCA for line scans / series; "
            "spatial disk-median for maps. More accurate when CRs stand out "
            "against neighbours, but may produce false positives on unusual "
            "spectral features."
        ),
    )
    force_1d = engine == CRR_ENGINE_1D

    with st.expander("1D engine parameters", expanded=True):
        col1, col2 = st.columns(2)
        spike_width = col1.number_input(
            "Spike width (channels, odd)",
            value=5,
            min_value=3,
            step=2,
            key="crr_spike_width",
            help=(
                "Widest cosmic-ray spike to look for, in spectral channels "
                "(median-filter window; must be odd). Widen it if broad "
                "spikes survive; narrow it if sharp real peaks get clipped."
            ),
        )
        if spike_width % 2 == 0:
            spike_width += 1
        spike_threshold = col2.number_input(
            "Detection threshold",
            value=3.5,
            min_value=0.1,
            step=0.5,
            key="crr_spike_threshold",
            help=(
                "How far above the noise level a point must stick out to "
                "count as a spike (threshold × estimated noise). Lower = "
                "more aggressive; raise it if real peaks are being removed."
            ),
        )
        spike_passes = st.number_input(
            "Passes",
            value=3,
            min_value=1,
            step=1,
            key="crr_spike_passes",
            help=(
                "How many times detection + repair is repeated per spectrum. "
                "Extra passes catch spikes that were masked by a larger "
                "neighbouring spike on the first pass."
            ),
        )

    # Default values used when the 2D/3D expander is hidden
    map_sensitivity = 0.01
    map_disk_radius = 3
    map_spike_width = 5
    map_method = "median"
    map_n_components = 3

    if not force_1d:
        with st.expander("Collection / 3D engine parameters"):
            c1, c2 = st.columns(2)
            map_sensitivity = c1.number_input(
                "Sensitivity",
                value=0.01,
                min_value=1e-4,
                step=0.005,
                format="%.4f",
                key="crr_map_sensitivity",
                help=(
                    "How different a pixel may be from its reference before "
                    "it is flagged as a cosmic ray. Lower = more aggressive "
                    "(more points flagged); raise it if real features are "
                    "being removed."
                ),
            )
            map_disk_radius = c2.number_input(
                "Neighbourhood radius (pixels)",
                value=3,
                min_value=1,
                step=1,
                key="crr_map_disk_radius",
                help=(
                    "For maps: how far around each pixel the spatial "
                    "reference (median of neighbours) reaches. Larger = "
                    "smoother reference, but slower and less local."
                ),
            )
            map_spike_width = c1.number_input(
                "Spike width (channels)",
                value=5,
                min_value=1,
                step=1,
                key="crr_map_spike_width",
                help=(
                    "Widest cosmic-ray spike to repair, in spectral "
                    "channels — same meaning as in the 1D engine."
                ),
            )
            map_method = c2.selectbox(
                "Reference method",
                ["median", "pca"],
                format_func={"median": "Median", "pca": "PCA"}.__getitem__,
                key="crr_map_method",
                help=(
                    "How the clean reference for line scans / series is "
                    "built: **Median** of all spectra (robust default) or a "
                    "low-rank **PCA** reconstruction (better when spectra "
                    "vary strongly across the scan)."
                ),
            )
            map_n_components = st.number_input(
                "PCA components",
                value=3,
                min_value=1,
                step=1,
                key="crr_map_n_components",
                help=(
                    "Only used with the PCA reference method: how many "
                    "principal components the reference keeps. More "
                    "components follow real variation more closely but may "
                    "start reproducing the spikes themselves."
                ),
            )

    return dict(
        force_1d=force_1d,
        spike_width=int(spike_width),
        spike_threshold=float(spike_threshold),
        spike_passes=int(spike_passes),
        map_sensitivity=float(map_sensitivity),
        map_disk_radius=int(map_disk_radius),
        map_spike_width=int(map_spike_width),
        map_method=map_method,
        map_n_components=int(map_n_components),
    )


def render_denoising_params() -> dict[str, Any]:
    """Render Denoiser parameter widgets. Returns denoiser params dict."""
    engine = st.selectbox(
        "Engine",
        options=[DENOISE_ENGINE_PCA, DENOISE_ENGINE_SMOOTHER],
        key="denoise_engine",
        help=(
            "**PCA — population-based:** fits a low-rank PCA model on all "
            "spectra together, then reconstructs each spectrum from the "
            "principal components that capture shared signal. Best for "
            "line scans and maps (≥ 2 spectra required).\n\n"
            "**Smoother — per spectrum:** applies a spectral smoothing filter "
            "(Savitzky-Golay or Whittaker) independently to every spectrum. "
            "Works on any data shape including single spectra."
        ),
    )
    per_spectrum = engine == DENOISE_ENGINE_SMOOTHER

    nc_type = "mle"
    nc_int = 2
    subtract_min = True
    restore_min = False

    if not per_spectrum:
        with st.expander("PCA parameters", expanded=True):
            # Older sessions may still hold the removed "float" / "None"
            # choices in widget state — fall back to automatic.
            if st.session_state.get("denoise_nc_type") not in (None, "mle", "int"):
                st.session_state["denoise_nc_type"] = "mle"
            nc_type = st.selectbox(
                "Number of components",
                ["mle", "int"],
                format_func={
                    "mle": "Automatic",
                    "int": "Fixed count",
                }.__getitem__,
                key="denoise_nc_type",
                help=(
                    "**Automatic** (recommended): estimates how many components "
                    "carry real signal (Minka's MLE).\n\n"
                    "**Fixed count**: keep exactly this many components — use "
                    "when you already know how many distinct spectral shapes "
                    "the dataset contains."
                ),
            )
            if nc_type == "int":
                nc_int = st.number_input(
                    "Component count",
                    value=2,
                    min_value=1,
                    step=1,
                    key="denoise_nc_int",
                    help=(
                        "How many principal components to keep. Too few "
                        "flattens real features; too many lets noise back in."
                    ),
                )
            baseline = st.selectbox(
                "Baseline handling",
                ["shape", "preserve", "raw"],
                format_func={
                    "shape": "Shape only (default)",
                    "preserve": "Preserve absolute intensities",
                    "raw": "Fit raw signal",
                }.__getitem__,
                key="denoise_baseline",
                help=(
                    "**Shape only** (default): subtracts each spectrum's minimum "
                    "before PCA so the model captures spectral shape, not baseline "
                    "offset. Output baseline is near zero. Best for most PL datasets.\n\n"
                    "**Preserve absolute intensities**: same subtraction before PCA, "
                    "but the saved minimum is added back afterward. Use when "
                    "downstream steps depend on absolute signal levels.\n\n"
                    "**Fit raw signal**: PCA sees the full signal including any "
                    "baseline offset. Useful when spectra have very different "
                    "baselines that should be part of the model."
                ),
            )
            subtract_min = baseline in ("shape", "preserve")
            restore_min = baseline == "preserve"

    smoother_params: dict[str, Any] = {}
    if per_spectrum:
        with st.expander("Smoother parameters", expanded=True):
            sm_method = st.selectbox(
                "Method",
                ["savgol", "whittaker", "wavelet"],
                format_func={
                    "savgol": "Savitzky-Golay",
                    "whittaker": "Whittaker-Eilers",
                    "wavelet": "Wavelet (VisuShrink)",
                }.__getitem__,
                key="denoise_sm_method",
                help=(
                    "**Savitzky-Golay**: fits a polynomial to a sliding window. "
                    "Fast and intuitive; good general-purpose smoother.\n\n"
                    "**Whittaker-Eilers**: penalised least squares — balances "
                    "fidelity to data vs. smoothness. λ controls the trade-off "
                    "(auto-selected by GCV if left on automatic).\n\n"
                    "**Wavelet (VisuShrink)**: decomposes the spectrum into "
                    "multi-scale components and zeros out noise-level coefficients. "
                    "Preserves sharp peaks without position shift — best choice "
                    "for PL spectra with narrow emission lines."
                ),
            )
            if sm_method == "savgol":
                sc1, sc2 = st.columns(2)
                sm_wl = sc1.number_input(
                    "Window size (points, odd)",
                    value=11,
                    min_value=3,
                    step=2,
                    key="denoise_sm_window_length",
                    help=(
                        "How many neighbouring points each polynomial fit "
                        "spans (must be odd). Larger = smoother but risks "
                        "flattening narrow peaks."
                    ),
                )
                if sm_wl % 2 == 0:
                    sm_wl += 1
                sm_po = sc2.number_input(
                    "Polynomial order",
                    value=3,
                    min_value=1,
                    max_value=int(sm_wl) - 1,
                    step=1,
                    key="denoise_sm_polyorder",
                    help=(
                        "Order of the polynomial fitted inside each window. "
                        "Higher preserves peak shape better but smooths less."
                    ),
                )
                smoother_params = dict(
                    method="savgol",
                    window_length=int(sm_wl),
                    polyorder=int(sm_po),
                    lam=None,
                    d=2,
                    auto_lam_calls=5,
                    wavelet="db4",
                    wavelet_level=None,
                    wavelet_threshold="soft",
                )
            elif sm_method == "whittaker":
                use_auto_lam = st.checkbox(
                    "Automatic smoothness (recommended)",
                    value=True,
                    key="denoise_sm_auto_lam",
                    help=(
                        "Picks the smoothness λ automatically by generalised "
                        "cross-validation (GCV). Untick to set λ yourself."
                    ),
                )
                sc1, sc2 = st.columns(2)
                sm_lam: float | None = None
                sm_alc = 5
                if not use_auto_lam:
                    sm_lam = sc1.number_input(
                        "Smoothness (λ)",
                        value=100.0,
                        min_value=0.001,
                        step=10.0,
                        key="denoise_sm_lam",
                        help=(
                            "Trade-off between following the data and "
                            "smoothness. Larger = smoother; spans orders of "
                            "magnitude, so try ×10 / ÷10 steps."
                        ),
                    )
                else:
                    sm_alc = st.number_input(
                        "Search steps for automatic λ",
                        value=5,
                        min_value=1,
                        step=1,
                        key="denoise_sm_auto_lam_calls",
                        help=(
                            "How many λ values the automatic search tries. "
                            "More steps = slightly better λ, slower run — 5 "
                            "is enough for most data."
                        ),
                    )
                sm_d = sc2.number_input(
                    "Difference order",
                    value=2,
                    min_value=1,
                    step=1,
                    key="denoise_sm_d",
                    help=(
                        "Order of the smoothness penalty. 2 (default) "
                        "penalises curvature and suits nearly all spectra."
                    ),
                )
                smoother_params = dict(
                    method="whittaker",
                    window_length=11,
                    polyorder=3,
                    lam=sm_lam,
                    d=int(sm_d),
                    auto_lam_calls=int(sm_alc),
                    wavelet="db4",
                    wavelet_level=None,
                    wavelet_threshold="soft",
                )
            else:  # wavelet
                wv_col1, wv_col2 = st.columns(2)
                wv_family = wv_col1.selectbox(
                    "Wavelet family",
                    ["db4", "db8", "sym4", "sym8", "coif3"],
                    key="denoise_sm_wavelet",
                    help=(
                        "Daubechies (db) and Symlets (sym) are general-purpose. "
                        "Higher numbers = wider support, smoother reconstruction. "
                        "Coiflets (coif) have extra vanishing moments — good for "
                        "spectra with polynomial baselines."
                    ),
                )
                wv_mode = wv_col2.selectbox(
                    "Threshold mode",
                    ["soft", "hard"],
                    key="denoise_sm_wavelet_threshold",
                    help=(
                        "**Soft**: shrinks coefficients toward zero continuously — "
                        "smoother output, slight amplitude reduction.\n\n"
                        "**Hard**: zeroes coefficients below the threshold — "
                        "preserves peak amplitudes, may leave more residual noise."
                    ),
                )
                wv_level = st.number_input(
                    "Decomposition level (0 = auto)",
                    value=0,
                    min_value=0,
                    step=1,
                    key="denoise_sm_wavelet_level",
                    help="0 uses the maximum level allowed by the signal length.",
                )
                smoother_params = dict(
                    method="wavelet",
                    window_length=11,
                    polyorder=3,
                    lam=None,
                    d=2,
                    auto_lam_calls=5,
                    wavelet=wv_family,
                    wavelet_level=None if int(wv_level) == 0 else int(wv_level),
                    wavelet_threshold=wv_mode,
                )

    return dict(
        n_components_type=nc_type,
        n_components_int=nc_int,
        subtract_min=subtract_min,
        restore_min=restore_min,
        per_spectrum=per_spectrum,
        smoother=smoother_params,
    )


def render_nmf_params() -> dict[str, Any]:
    """Render advanced NMF parameter widgets. Returns nmf_params dict.

    Deliberately excludes ``n_components`` — that is chosen on the
    Decomposition page itself from the diagnostic curve, not a plain
    number_input here, since the whole point of the diagnostic curve is to
    inform that choice interactively rather than default to a hidden value.
    """
    with st.expander("Advanced NMF parameters", expanded=False):
        init = st.selectbox(
            "Initialization",
            ["nndsvda", "nndsvd", "random"],
            key="nmf_init",
            help=(
                "**nndsvda** (default): deterministic SVD-based init that "
                "fills exact zeros with the data average. Fast and "
                "reproducible — recommended.\n\n"
                "**nndsvd**: deterministic SVD-based init with exact zeros; "
                "can stall on sparse components.\n\n"
                "**random**: random non-negative init seeded by "
                "random_state. Useful for checking whether a result is "
                "stable across different starting points."
            ),
        )
        col1, col2 = st.columns(2)
        max_iter = col1.number_input(
            "Max iterations",
            value=500,
            min_value=50,
            step=50,
            key="nmf_max_iter",
            help=(
                "Hard cap on optimisation iterations. Raise it if the fit reports non-convergence."
            ),
        )
        random_state = col2.number_input(
            "Random seed",
            value=0,
            min_value=0,
            step=1,
            key="nmf_random_state",
            help=(
                "Seed for the random initialisation — keeps repeated runs "
                "reproducible. Only matters with random init."
            ),
        )
    return dict(
        init=init,
        max_iter=int(max_iter),
        random_state=int(random_state),
    )


def render_mcr_params() -> dict[str, Any]:
    """Render advanced MCR-ALS solver widgets. Returns an mcr_params dict.

    Every knob here has a sensible default and stays collapsed, so a chemist
    can run MCR-ALS without touching any of it — the physical choices (how
    many components, whether to pin a reference) live on the page itself, not
    in this expander.

    Deliberately excludes ``n_components`` (chosen on the page from the SVD
    scree) and the equality reference (a page-level, physical choice).
    """
    with st.expander("Advanced MCR-ALS parameters", expanded=False):
        col1, col2 = st.columns(2)
        max_iter = col1.number_input(
            "Max iterations",
            value=200,
            min_value=20,
            step=20,
            key="mcr_max_iter",
            help=(
                "Hard cap on ALS iterations. The fit normally stops earlier "
                "when the change in lack-of-fit falls below the convergence "
                "threshold."
            ),
        )
        tol = col2.number_input(
            "Convergence threshold (%ΔLOF)",
            value=0.1,
            min_value=0.001,
            max_value=5.0,
            step=0.05,
            format="%.3f",
            key="mcr_tol",
            help=(
                "Stop when the lack-of-fit (%LOF) changes by less than this "
                "between iterations. 0.1% is the common default."
            ),
        )
        col3, col4 = st.columns(2)
        simplisma_offset = col3.number_input(
            "SIMPLISMA noise offset (%)",
            value=5.0,
            min_value=0.1,
            max_value=50.0,
            step=1.0,
            key="mcr_offset",
            help=(
                "Noise floor for the pure-pixel search that seeds the fit. "
                "Higher values are more robust to noisy spectra; 5% is a good "
                "default."
            ),
        )
        random_state = col4.number_input(
            "Random seed",
            value=0,
            min_value=0,
            step=1,
            key="mcr_random_state",
            help="Seed for subsampling in the rank/ambiguity diagnostics.",
        )
    return dict(
        max_iter=int(max_iter),
        tol=float(tol),
        simplisma_offset=float(simplisma_offset) / 100.0,
        random_state=int(random_state),
    )

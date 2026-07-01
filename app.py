"""WDF Viewer — entry point."""

import sys
from pathlib import Path

# Lets backend sub-packages use plain imports (e.g. `from cosmic_ray import ...`)
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import streamlit as st

from frontend.sidebar import render_sidebar
from frontend.pages.data_overview import render_data_page
from frontend.pages.preprocessing import render_preprocessing_page
from frontend.pages.decomposition import render_decomposition_page
from frontend.pages.deconvolution import render_deconvolution_page
from frontend.pages.map_analysis import render_map_page

st.set_page_config(
    page_title="SpectraLab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="column"]:first-child { padding-right: 1rem; }
    /* --- Typography Tier 1: block / section titles --- */
    .section-header {
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #555;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }
    /* --- Typography Tier 3: metadata, file info --- */
    .info-box {
        background: #f4f6fa;
        border-radius: 6px;
        padding: 0.45rem 0.7rem;
        font-family: monospace;
        font-size: 0.80rem;
        line-height: 1.65;
        margin-bottom: 0.4rem;
    }
    .stExpander { border: 1px solid #e0e4ec !important; border-radius: 6px !important; }
    .st-key-remove_files button {
        background-color: #fee2e2;
        border-color: #fca5a5;
        color: #991b1b;
    }
    .st-key-remove_files button:hover {
        background-color: #fecaca;
        border-color: #f87171;
        color: #7f1d1d;
    }
    /* Step bar — Tier 1 typography to match section headers */
    div.st-key-step_bar a {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Named page objects — referenced by _render_step_bar for st.switch_page
# ---------------------------------------------------------------------------

_page_data           = st.Page(render_data_page,            title="Data",
                                icon=":material/description:", default=True)
_page_preprocessing  = st.Page(render_preprocessing_page,  title="Preprocessing",
                                icon=":material/tune:")
_page_decomposition  = st.Page(render_decomposition_page,  title="Decomposition",
                                icon=":material/category:")
_page_deconvolution  = st.Page(render_deconvolution_page,  title="Deconvolution",
                                icon=":material/timeline:")
_page_map            = st.Page(render_map_page,             title="Map Analysis",
                                icon=":material/map:")

pg = st.navigation(
    [_page_data, _page_preprocessing, _page_decomposition, _page_deconvolution, _page_map],
    position="hidden",
)


def _render_step_bar(pg) -> None:
    """Horizontal step workflow indicator rendered above page content.

    All steps are rendered as st.page_link so they share a consistent
    appearance. The active step is highlighted via injected CSS targeting
    its container key — avoiding the jarring button-vs-link visual switch
    that occurred with the old disabled-button approach.
    """
    _steps = [
        (_page_data,          "Data"),
        (_page_preprocessing, "Preprocessing"),
        (_page_decomposition, "Decomposition"),
        (_page_deconvolution, "Deconvolution"),
        (_page_map,           "Map Analysis"),
    ]
    current_idx = next((i for i, (page, _) in enumerate(_steps) if page is pg), 0)
    st.markdown(
        f"<style>div.st-key-nav_item_{current_idx} a "
        f"{{ background: rgba(49,51,63,0.08) !important; }}</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="step_bar"):
        cols = st.columns(len(_steps), gap="small")
        for i, (col, (page, title)) in enumerate(zip(cols, _steps)):
            label = f"{i + 1}. {title}"
            with col:
                with st.container(key=f"nav_item_{i}"):
                    st.page_link(page, label=label, use_container_width=True)
    st.divider()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

with st.sidebar:
    render_sidebar()

_render_step_bar(pg)
pg.run()

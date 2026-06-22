"""WDF Viewer — entry point."""

import sys
from pathlib import Path

# Lets backend sub-packages use plain imports (e.g. `from cosmic_ray import ...`)
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import streamlit as st

from frontend.sidebar import render_sidebar
from frontend.pages.preprocessing import render_preprocessing_page
from frontend.pages.map_analysis import render_map_page
from frontend.pages.deconvolution import render_deconvolution_page

st.set_page_config(
    page_title="SpectraLab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="column"]:first-child { padding-right: 1rem; }
    .section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #555;
        margin-top: 0.8rem;
        margin-bottom: 0.2rem;
    }
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    render_sidebar()

pg = st.navigation([
    st.Page(render_preprocessing_page, title="Preprocessing",
            icon=":material/tune:", default=True),
    st.Page(render_map_page,           title="Map Analysis",
            icon=":material/map:"),
    st.Page(render_deconvolution_page, title="Deconvolution",
            icon=":material/timeline:"),
])
pg.run()

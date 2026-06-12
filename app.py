"""WDF Viewer — entry point."""

import sys
from pathlib import Path

# Lets backend sub-packages use plain imports (e.g. `from cosmic_ray import ...`)
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import streamlit as st

from frontend.left_panel import render_left_panel
from frontend.tabs import render_tabs

st.set_page_config(
    page_title="SpectraLab",
    layout="wide",
    initial_sidebar_state="collapsed",
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

left, right = st.columns([1, 2], gap="medium")

with left:
    state = render_left_panel()

with right:
    render_tabs(state)

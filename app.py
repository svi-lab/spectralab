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


@st.cache_resource
def _app_stylesheet() -> str:
    return (Path(__file__).parent / "frontend" / "style.css").read_text()


st.markdown(f"<style>{_app_stylesheet()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Named page objects — referenced by _render_step_bar for st.switch_page
# ---------------------------------------------------------------------------

_page_data = st.Page(render_data_page, title="Data", icon=":material/description:", default=True)
_page_preprocessing = st.Page(
    render_preprocessing_page, title="Preprocessing", icon=":material/tune:"
)
_page_decomposition = st.Page(
    render_decomposition_page, title="Decomposition", icon=":material/category:"
)
_page_deconvolution = st.Page(
    render_deconvolution_page, title="Deconvolution", icon=":material/timeline:"
)
_page_map = st.Page(render_map_page, title="Map Analysis", icon=":material/map:")

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
        (_page_data, "Data"),
        (_page_preprocessing, "Preprocessing"),
        (_page_decomposition, "Decomposition"),
        (_page_deconvolution, "Deconvolution"),
        (_page_map, "Map Analysis"),
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
    has_files = render_sidebar()

if not has_files:
    st.markdown(
        '<div class="sl-hero">'
        '<h1 class="sl-hero-title">SpectraLab</h1>'
        '<p class="sl-hero-subtitle">&larr; Upload a file in the sidebar to start analysis</p>'
        "</div>",
        unsafe_allow_html=True,
    )
else:
    _render_step_bar(pg)
    pg.run()

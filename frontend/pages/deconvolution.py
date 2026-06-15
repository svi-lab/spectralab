# -*- coding: utf-8 -*-
"""Gaussian deconvolution page (placeholder)."""

from __future__ import annotations

import streamlit as st


def render_deconvolution_page() -> None:
    """Placeholder for the Gaussian deconvolution page."""
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.markdown('<p class="section-header">Peak Parameters</p>', unsafe_allow_html=True)
        st.info("Coming soon — peak fitting controls will appear here.")
    with right:
        st.info("Coming soon — Gaussian deconvolution chart will appear here.")

# -*- coding: utf-8 -*-
"""Cached pipeline accessor — single entry point used by every page.

All four pages need the processed (preprocess()'d) DataArray per loaded file.
Routing them all through the same st.cache_data-wrapped call means:
  - the heavy CRR/Denoiser/Normalize work only ever runs once per
    (file content, pipeline params) combination, no matter which page
    triggers it first;
  - a file added after another page already has results never goes stale,
    since cache lookups happen on every rerun instead of a one-shot
    "compute once into session_state" gate.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.pipeline import preprocess
from backend._shared.dataset import SpectralDataset


def default_pipeline_params() -> dict[str, Any]:
    """All-disabled pipeline params for pages opened before Preprocessing has run."""
    return {
        "norm1_enabled": False, "norm1": {},
        "cd_enabled":    False, "cd":    {},
        "crr_enabled":   False, "crr":   {},
        "norm2_enabled": False, "norm2": {},
        "denoise_enabled": False, "denoise": {},
        "norm3_enabled": False, "norm3": {},
        "bg_enabled":    False, "bg":    {},
    }


@st.cache_data(show_spinner=False, max_entries=16)
def _preprocess_cached(
    file_hash: str, _dataset: SpectralDataset, pipeline_params: dict
) -> tuple[dict, Any]:
    return preprocess(_dataset, pipeline_params)


def get_finals(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any] | None = None,
) -> tuple[dict[str, dict], dict[str, Any], list[str]]:
    """Run the cached pipeline for every loaded file.

    Returns (all_stages, all_finals, errors) — same shape regardless of
    which page calls it first.
    """
    params = pipeline_params or default_pipeline_params()
    all_stages: dict[str, dict] = {}
    all_finals: dict[str, Any] = {}
    errors: list[str] = []
    for name, entry in loaded.items():
        try:
            stages, da_final = _preprocess_cached(entry["hash"], entry["dataset"], params)
            all_stages[name] = stages
            all_finals[name] = da_final
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return all_stages, all_finals, errors

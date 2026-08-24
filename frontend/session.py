# -*- coding: utf-8 -*-
"""Session-state keys that must reset when the uploaded file set changes.

Kept as a plain module so tests can exercise the list without rendering the
sidebar. Widget keys listed here are popped so a re-upload cannot inherit
the previous file's NMF / MCR / deconv results.
"""

from __future__ import annotations

from typing import MutableMapping

# Results and pipeline blobs. Leaving any of these behind after "Remove all
# files" would show the previous file's fit / decomposition on the next upload.
ANALYSIS_KEYS: tuple[str, ...] = (
    "sl_pipeline_params",
    "sl_sample_structure",
    "sl_excluded",
    "sl_excluded_undo",
    "_excl_last_selection",
    "_sl_finals_memo",
    "sl_nmf_result",
    "sl_nmf_diagnostic",
    "sl_mcr_result",
    "sl_mcr_rank",
    "sl_deconv_result",
    "sl_deconv_result_file",
    "sl_deconv_result_digest",
    "sl_deconv_batch_result",
    "sl_deconv_batch_labels",
    "sl_deconv_batch_file",
    "deconv_bands_source",
    "deconv_bands_table",
    "deconv_bands_rev",
    "deconv_bands_unit",
)

# Widget keys that survive a page remount if not popped. Includes leftover
# background-suppression keys from older sessions.
WIDGET_KEYS: tuple[str, ...] = (
    "cd_enabled", "crr_enabled", "denoise_enabled", "norm_selection", "prog_title",
    "excl_file", "excl_mode", "excl_rows", "excl_cols", "excl_pixels", "excl_flat",
    "bg_enabled", "bg_ref_source", "bg_ref_file",
    "bg_row_min", "bg_row_max", "bg_col_min", "bg_col_max",
    "bg_pt_ratio", "bg_c_override_on", "bg_c_override",
    "sl_bg_ui",
)


def clear_analysis_state(ss: MutableMapping) -> None:
    """Drop pipeline, exclusion, memo, and analysis-result keys from ``ss``."""
    for key in ANALYSIS_KEYS:
        ss.pop(key, None)
    for key in WIDGET_KEYS:
        ss.pop(key, None)

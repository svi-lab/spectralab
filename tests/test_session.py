"""Remove-all-files must drop analysis results, not just the upload list."""

from __future__ import annotations

from frontend.session import ANALYSIS_KEYS, clear_analysis_state


def test_clear_analysis_state_drops_decomp_and_deconv_results():
    ss = {
        "sl_nmf_result": {"file_name": "old.wdf"},
        "sl_mcr_result": {"file_name": "old.wdf"},
        "sl_deconv_result": object(),
        "sl_deconv_batch_result": object(),
        "_sl_finals_memo": {("h", "d", False): object()},
        "sl_pipeline_params": {"crr_enabled": True},
        "sl_loaded": {"keep": True},
        "_sl_uploader_key": 3,
    }
    clear_analysis_state(ss)
    for key in ANALYSIS_KEYS:
        assert key not in ss
    assert ss["sl_loaded"] == {"keep": True}
    assert ss["_sl_uploader_key"] == 3


def test_clear_analysis_state_is_idempotent():
    ss: dict = {}
    clear_analysis_state(ss)
    assert ss == {}

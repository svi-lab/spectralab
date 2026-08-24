"""Finals memo identity, digest skipping exclusion/bg leftovers, mask overlay."""

from __future__ import annotations

import numpy as np
import pytest

from frontend.pipeline_cache import (
    _DIGEST_SKIP,
    _params_digest,
    default_pipeline_params,
    final_da,
    get_finals,
)
from tests.factories import gaussian_map, make_map


@pytest.fixture
def session(monkeypatch):
    """Plain dict standing in for st.session_state (memo lives there)."""
    ss: dict = {}
    monkeypatch.setattr("frontend.pipeline_cache.st.session_state", ss)
    return ss


def _loaded(name: str = "a.wdf", file_hash: str = "h1"):
    return {name: {"hash": file_hash, "dataset": make_map(gaussian_map())}}


def test_digest_ignores_exclusion_masks():
    p1 = default_pipeline_params()
    p1["excl"] = {"masks": {"a.wdf": np.ones((2, 2), dtype=bool)}}
    p2 = default_pipeline_params()
    p2["excl"] = {"masks": {"b.wdf": np.zeros((9, 9), dtype=bool)}}
    d1 = _params_digest({k: v for k, v in p1.items() if k not in _DIGEST_SKIP})
    d2 = _params_digest({k: v for k, v in p2.items() if k not in _DIGEST_SKIP})
    assert d1 == d2


def test_digest_ignores_legacy_bg_keys():
    p = default_pipeline_params()
    base = _params_digest({k: v for k, v in p.items() if k not in _DIGEST_SKIP})
    p["bg_enabled"] = False
    p["bg"] = {"reference": np.arange(64.0)}
    leftover = _params_digest({k: v for k, v in p.items() if k not in _DIGEST_SKIP})
    assert leftover == base


def test_digest_changes_when_a_real_stage_flips():
    p = default_pipeline_params()
    before = _params_digest({k: v for k, v in p.items() if k not in _DIGEST_SKIP})
    p["cd_enabled"] = True
    p["cd"] = {"n_zeros": 10}
    after = _params_digest({k: v for k, v in p.items() if k not in _DIGEST_SKIP})
    assert before != after


def test_get_finals_same_params_returns_same_object(session):
    loaded = _loaded()
    params = default_pipeline_params()
    d1, e1 = get_finals(loaded, params)
    d2, e2 = get_finals(loaded, params)
    assert e1 == e2 == []
    assert d1["a.wdf"] is d2["a.wdf"]


def test_mask_edit_builds_on_pre_exclusion_memo(session):
    loaded = _loaded()
    params = default_pipeline_params()
    params["norm1_enabled"] = True
    params["norm1"] = {"method": "min_max"}

    unmasked, _ = get_finals(loaded, params, keep_stages=False)
    pre = unmasked["a.wdf"]

    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 1] = True
    params["excl"] = {"masks": {"a.wdf": mask}}
    masked, errors = get_finals(loaded, params, keep_stages=False)
    assert errors == []
    ds = masked["a.wdf"]
    assert ds is not pre
    assert ds.attrs["final_var"] == "excluded"
    assert np.isnan(final_da(ds).values[0, 1]).all()

    # Pre-exclusion entry is still the same object — mask edit must not
    # rebuild the stage chain.
    again_unmasked, _ = get_finals(
        loaded,
        {**params, "excl": {}},
        keep_stages=False,
    )
    assert again_unmasked["a.wdf"] is pre

    # Same mask → memo identity.
    masked2, _ = get_finals(loaded, params, keep_stages=False)
    assert masked2["a.wdf"] is ds


def test_keep_stages_false_reuses_true_entry_without_copying_values(session):
    loaded = _loaded()
    params = default_pipeline_params()
    params["norm1_enabled"] = True
    params["norm1"] = {"method": "min_max"}
    full, _ = get_finals(loaded, params, keep_stages=True)
    final_only, _ = get_finals(loaded, params, keep_stages=False)
    fv = full["a.wdf"].attrs["final_var"]
    assert fv == final_only["a.wdf"].attrs["final_var"]
    assert full["a.wdf"][fv].values is final_only["a.wdf"][fv].values


def test_two_files_mask_on_one_does_not_touch_the_other(session):
    ds_a = make_map(gaussian_map())
    ds_b = make_map(gaussian_map())
    loaded = {
        "a.wdf": {"hash": "ha", "dataset": ds_a},
        "b.wdf": {"hash": "hb", "dataset": ds_b},
    }
    params = default_pipeline_params()
    first, _ = get_finals(loaded, params)
    obj_b = first["b.wdf"]

    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True
    params["excl"] = {"masks": {"a.wdf": mask}}
    second, _ = get_finals(loaded, params)
    assert second["b.wdf"] is obj_b
    assert second["a.wdf"].attrs["final_var"] == "excluded"

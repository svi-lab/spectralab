"""Pipeline sequencing, exclusion shape/dtype, and preprocess vs run_stage_chain."""

from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline import (
    apply_exclusion,
    preprocess,
    run_stage_chain,
    stage_exclude,
    stage_normalize,
)
from frontend.pipeline_cache import default_pipeline_params
from tests.factories import gaussian_map, make_map


def test_disabled_pipeline_is_raw():
    ds = make_map(gaussian_map())
    out = preprocess(ds, default_pipeline_params(), keep_stages=True)
    assert out.attrs["final_var"] == "raw"
    assert out.attrs["stage_vars"] == ["raw"]
    np.testing.assert_array_equal(out["raw"].values, ds.da.values)


def test_minmax_norm_then_exclude_preserves_shape_and_dtype():
    values = gaussian_map(dtype=np.float32)
    ds = make_map(values)
    params = default_pipeline_params()
    params["norm1_enabled"] = True
    params["norm1"] = {"method": "min_max"}
    mask = np.zeros(values.shape[:2], dtype=bool)
    mask[0, 0] = True
    params["excl"] = {"mask": mask}

    out = preprocess(ds, params, keep_stages=True)
    assert out.attrs["final_var"] == "excluded"
    assert "norm_before" in out.data_vars
    final = out["excluded"]
    assert final.shape == ds.da.shape
    assert final.dtype == np.float32
    assert np.isnan(final.values[0, 0]).all()
    assert np.isfinite(final.values[1, 0]).all()


def test_stage_exclude_rejects_wrong_shape():
    ds = make_map(gaussian_map())
    with pytest.raises(ValueError, match="shape"):
        stage_exclude(ds.da, np.zeros((2, 2), dtype=bool))


def test_stage_exclude_empty_mask_returns_same_object():
    ds = make_map(gaussian_map())
    empty = np.zeros(ds.da.shape[:2], dtype=bool)
    assert stage_exclude(ds.da, empty) is ds.da


def test_stage_exclude_does_not_mutate_input():
    ds = make_map(gaussian_map())
    original = ds.da.values.copy()
    mask = np.zeros(ds.da.shape[:2], dtype=bool)
    mask[2, 1] = True
    out = stage_exclude(ds.da, mask)
    np.testing.assert_array_equal(ds.da.values, original)
    assert np.isnan(out.values[2, 1]).all()


def test_normalize_restores_float32():
    ds = make_map(gaussian_map(dtype=np.float32))
    out = stage_normalize(ds.da, {"method": "min_max"})
    assert out.dtype == np.float32
    assert float(np.nanmax(out.values)) == pytest.approx(1.0, abs=1e-5)


def test_preprocess_and_run_stage_chain_match_without_mask():
    ds = make_map(gaussian_map())
    params = default_pipeline_params()
    params["norm1_enabled"] = True
    params["norm1"] = {"method": "area"}
    a = preprocess(ds, params, keep_stages=True)
    b = run_stage_chain(ds, params, keep_stages=True)
    assert a.attrs["stage_vars"] == b.attrs["stage_vars"]
    np.testing.assert_allclose(
        a[a.attrs["final_var"]].values,
        b[b.attrs["final_var"]].values,
    )


def test_apply_exclusion_on_keep_stages_false_only_stores_final():
    ds = make_map(gaussian_map())
    params = default_pipeline_params()
    params["norm1_enabled"] = True
    params["norm1"] = {"method": "min_max"}
    pre = run_stage_chain(ds, params, keep_stages=False)
    mask = np.zeros(ds.da.shape[:2], dtype=bool)
    mask[-1, -1] = True
    out = apply_exclusion(pre, mask, ds.spectral_dim, keep_stages=False)
    assert list(out.data_vars) == ["excluded"]
    assert out.attrs["final_var"] == "excluded"
    assert "norm_before" in out.attrs["stage_vars"]

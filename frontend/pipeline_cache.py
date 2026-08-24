# -*- coding: utf-8 -*-
"""Cached pipeline accessor — single entry point used by every page.

All five pages need the processed DataArray per loaded file. Only the two
stages that are actually expensive to recompute — Cosmic Ray Removal and
Denoising (PCA population fit) — get their own `@st.cache_data` entry, keyed
on the cumulative "recipe so far" so a downstream param change never
recomputes them. Normalize / CleanData are cheap vectorized numpy and run
eagerly on every call instead.

Sequencing lives in ``backend.pipeline.run_stage_chain``. This module only
injects the cached CRR/denoise callables and the session-level finals memo.

`get_finals` iterates every loaded file on EVERY call (once per page per
Streamlit rerun), so with N files loaded, only the `max_entries` most-recent
(file, recipe) combos stay resident for CRR/denoise — if `max_entries` is
smaller than the number of loaded files, the cache thrashes and the
expensive stages recompute on literally every rerun instead of ever hitting.

The recipe dict grows stage by stage: each cached call's key contains every
upstream parameter plus the stage's own.

Session-level final-result memo
--------------------------------
On top of the two `st.cache_data` stage caches above sits a plain dict in
`st.session_state` (`_sl_finals_memo`), keyed on
``(file_hash, params_digest, keep_stages)``. It exists because
`st.cache_data` copies (unpickles) its return value on every single read —
for a same-params rerun (the overwhelmingly common case: the user is looking
at a chart, not touching a widget) that means every page load pays a full
pickle.loads of every upstream array even though nothing changed. The memo
returns the exact same object reference instead — zero copy, zero pickle.

The two layers serve different reuse patterns and both stay:
  - the memo serves same-params reruns (dominant case, <1 ms/file);
  - the `_crr_cached`/`_denoise_cached` st.cache_data entries serve
    cross-params reuse (e.g. toggling a later normalization after CRR
    already ran — new digest, memo miss, but CRR still hits its own cache).

Manual exclusion
----------------
The user's per-file exclusion mask (frontend/exclusion.py) arrives in
`params["excl"]["masks"]` but is deliberately kept OUT of the params digest.
Instead the digest is built from every *other* param, and the mask contributes
a separate `mask_digest` tag appended to that base:

    memo key = (file_hash, base_digest + mask_tag, keep_stages)

Two reasons. First, the digest stays per-file: editing one file's mask cannot
invalidate another file's memo entry (the params dict is global and shared by
every page). Second, `apply_exclusion` is applied *on top of* the memoized
pre-exclusion Dataset (key `base_digest`, mask_tag = "") rather than by
re-running the stage chain — so a mask edit never reaches `_crr_cached` /
`_denoise_cached`. Each edit costs exactly one `np.where` copy of the final
array, for the edited file only.

**Immutability contract**: memo'd Datasets are shared references. Every stage
function returns a *new* DataArray rather than mutating its input — no
consumer may write into a Dataset/DataArray it got from `get_finals`.
`apply_exclusion` copies the attrs dicts it extends rather than mutating the
pre-exclusion Dataset's.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import streamlit as st
import xarray as xr

from backend.pipeline import (
    apply_exclusion,
    run_stage_chain,
    stage_cosmic_ray_removal,
    stage_denoise,
)
from backend._shared.dataset import SpectralDataset
from .exclusion import mask_digest

# CRR/denoise cache budget. Each retained entry is a full-size array (~150 MB
# for a 2000-spectra x 9341-channel float64 map). Must be >= the number of
# files typically loaded at once, or every rerun thrashes and recomputes the
# expensive stages for files whose params never changed.
_STAGE_MAX_ENTRIES = 16

# Final-result memo (see module docstring). Separate caps for keep_stages —
# True entries hold every intermediate stage array (~4-6x the memory of a
# final-only entry) — but get_finals() is called once per loaded file every
# rerun, so the cap must still be >= a realistic number of simultaneously
# loaded files (same reasoning as _STAGE_MAX_ENTRIES above) or the
# keep_stages=True bucket — used by Preprocessing's Progress tab, the one
# page most likely to have several files loaded — evicts and recomputes in
# rotation within a single get_finals() call.
#
# A file with an exclusion mask occupies TWO entries per bucket (the
# mask-independent pre-exclusion result and the masked one), so the caps are
# 2x the target file count rather than 1x — otherwise masking a few files
# evicts the pre-exclusion entries they are built from, and every rerun pays
# a full stage-chain re-run to rebuild them.
_FINALS_MEMO_KEY = "_sl_finals_memo"
_FINALS_MEMO_CAPS: dict[bool, int] = {False: 16, True: 16}

# Keys that must not affect the shared params digest. ``excl`` is re-applied
# per file via mask_digest. ``bg`` / ``bg_enabled`` are leftover keys from
# older sessions — stripping them keeps a re-opened app on the same memo
# entries after those fields were removed from the live params dict.
_DIGEST_SKIP = frozenset({"excl", "bg", "bg_enabled"})


def default_pipeline_params() -> dict[str, Any]:
    """All-disabled pipeline params for pages opened before Preprocessing has run."""
    return {
        "norm1_enabled": False, "norm1": {},
        "cd_enabled":    False, "cd":    {},
        "crr_enabled":   False, "crr":   {},
        "norm2_enabled": False, "norm2": {},
        "denoise_enabled": False, "denoise": {},
        "norm3_enabled": False, "norm3": {},
        "excl":          {},
    }


# ---------------------------------------------------------------------------
# Cached calls — CRR and denoise only (see module docstring for why).
#
# `_da` (the upstream array) is excluded from hashing — `file_hash + recipe`
# fully determine it, since the recipe contains every upstream parameter.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=_STAGE_MAX_ENTRIES)
def _crr_cached(file_hash: str, _da: xr.DataArray, recipe: dict) -> xr.DataArray:
    return stage_cosmic_ray_removal(_da, recipe["crr"])


@st.cache_data(show_spinner=False, max_entries=_STAGE_MAX_ENTRIES)
def _denoise_cached(file_hash: str, _da: xr.DataArray, recipe: dict) -> xr.DataArray:
    return stage_denoise(_da, recipe["denoise"])


# ---------------------------------------------------------------------------
# Orchestrator — injects cached CRR/denoise into backend.run_stage_chain.
# ---------------------------------------------------------------------------

def _run_stage_chain(
    file_hash: str,
    dataset: SpectralDataset,
    params: dict[str, Any],
    keep_stages: bool,
) -> xr.Dataset:
    return run_stage_chain(
        dataset,
        params,
        keep_stages,
        cosmic_ray=lambda da, rec: _crr_cached(file_hash, da, rec),
        denoise=lambda da, rec: _denoise_cached(file_hash, da, rec),
    )


# ---------------------------------------------------------------------------
# Params digest — deterministic <1 ms fingerprint for the finals memo
# ---------------------------------------------------------------------------

def _digest_walk(obj: Any, h: "hashlib._Hash") -> None:
    """Feed a deterministic byte representation of ``obj`` into ``h``.

    Dict keys are sorted so key-insertion order never changes the digest;
    ndarrays are hashed by raw bytes+dtype+shape rather than ``repr``
    (which truncates/summarizes large arrays).
    """
    if isinstance(obj, dict):
        h.update(b"d")
        for key in sorted(obj.keys(), key=repr):
            h.update(repr(key).encode())
            _digest_walk(obj[key], h)
    elif isinstance(obj, (list, tuple)):
        h.update(b"l")
        for item in obj:
            _digest_walk(item, h)
    elif isinstance(obj, np.ndarray):
        h.update(b"a")
        h.update(str(obj.dtype).encode())
        h.update(str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    else:
        h.update(repr(obj).encode())


def _params_digest(params: dict[str, Any]) -> str:
    """Deterministic fingerprint of a pipeline params dict (<1 ms typical)."""
    h = hashlib.blake2b(digest_size=16)
    _digest_walk(params, h)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Finals memo — plain dict in st.session_state (see module docstring)
# ---------------------------------------------------------------------------

def _finals_memo() -> dict[tuple[str, str, bool], xr.Dataset]:
    return st.session_state.setdefault(_FINALS_MEMO_KEY, {})


def _memo_put(
    memo: dict[tuple[str, str, bool], xr.Dataset],
    key: tuple[str, str, bool],
    ds: xr.Dataset,
) -> None:
    """Insert or promote to most-recently-used, then evict oldest entries of
    the same keep_stages bucket over its cap (true insertion-order LRU: a
    dict pop+reinsert moves the key to the end even on a hit, no cost paid
    for tracking last-access beyond that)."""
    memo.pop(key, None)
    memo[key] = ds
    cap = _FINALS_MEMO_CAPS[key[2]]
    bucket = [k for k in memo if k[2] == key[2]]
    for stale in bucket[:-cap] if cap > 0 else bucket:
        memo.pop(stale, None)


def _final_only_view(full_ds: xr.Dataset) -> xr.Dataset:
    """Build a final-only Dataset from a keep_stages=True entry — shares the
    same underlying DataArray (no compute, no copy)."""
    final_var = full_ds.attrs["final_var"]
    ds = xr.Dataset({final_var: full_ds[final_var]})
    ds.attrs["stage_vars"] = full_ds.attrs["stage_vars"]
    ds.attrs["stage_labels"] = full_ds.attrs["stage_labels"]
    ds.attrs["final_var"] = final_var
    return ds


def _finals_for_file(
    entry: dict[str, Any],
    params: dict[str, Any],
    base_digest: str,
    mask: np.ndarray | None,
    keep_stages: bool,
    memo: dict[tuple[str, str, bool], xr.Dataset],
) -> xr.Dataset:
    """Resolve one file's pipeline Dataset, exclusion mask included."""
    file_hash = entry["hash"]
    dataset: SpectralDataset = entry["dataset"]
    mask_tag = mask_digest(mask)

    key = (file_hash, base_digest + mask_tag, keep_stages)
    cached = memo.get(key)
    if cached is not None:
        _memo_put(memo, key, cached)  # promote to most-recently-used
        return cached

    # Pre-exclusion result: the mask-independent key, which is exactly the key
    # this file had before any mask existed.
    pre_key = (file_hash, base_digest, keep_stages)
    ds_pre = memo.get(pre_key)
    if ds_pre is not None:
        _memo_put(memo, pre_key, ds_pre)
    else:
        if not keep_stages:
            full_ds = memo.get((file_hash, base_digest, True))
            if full_ds is not None:
                ds_pre = _final_only_view(full_ds)
        if ds_pre is None:
            ds_pre = _run_stage_chain(file_hash, dataset, params, keep_stages)
        _memo_put(memo, pre_key, ds_pre)

    if not mask_tag:
        return ds_pre

    ds = apply_exclusion(ds_pre, mask, dataset.spectral_dim, keep_stages)
    _memo_put(memo, key, ds)
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_finals(
    loaded: dict[str, Any],
    pipeline_params: dict[str, Any] | None = None,
    keep_stages: bool = False,
) -> tuple[dict[str, xr.Dataset], list[str]]:
    """Run the cached stage chain for every loaded file.

    Returns (all_datasets, errors) — one xr.Dataset per file. By default the
    Dataset holds only the final data variable (`final_da(ds)` to read it);
    pass ``keep_stages=True`` to also include every intermediate stage
    (Preprocessing page's Progress tab).

    A same-(file, params, keep_stages) call within the session is answered
    straight from the finals memo (see module docstring) — no stage code
    runs, not even a cache lookup for CRR/denoise.

    Manual exclusion masks (``params["excl"]["masks"]``) are applied on top of
    the memoized pre-exclusion result and are keyed separately per file, so a
    mask edit costs one array copy for the edited file and nothing else.
    """
    params = pipeline_params or default_pipeline_params()
    masks = (params.get("excl") or {}).get("masks") or {}
    # The mask is excluded from the shared digest and re-enters per file as a
    # key suffix — see the module docstring.
    base_digest = _params_digest({k: v for k, v in params.items() if k not in _DIGEST_SKIP})
    memo = _finals_memo()

    all_datasets: dict[str, xr.Dataset] = {}
    errors: list[str] = []
    for name, entry in loaded.items():
        try:
            all_datasets[name] = _finals_for_file(
                entry, params, base_digest, masks.get(name), keep_stages, memo,
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return all_datasets, errors


def final_da(ds: xr.Dataset | None) -> xr.DataArray | None:
    """The fully-processed final DataArray of a per-file pipeline Dataset."""
    if ds is None:
        return None
    return ds[ds.attrs["final_var"]]


def stage_dict(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Ordered label → DataArray mapping of the stages stored in ``ds``.

    Feeds charts.make_progress_echarts; only meaningful for Datasets built
    with ``keep_stages=True`` (otherwise it contains just the final stage).
    """
    return {
        ds.attrs["stage_labels"][var]: ds[var]
        for var in ds.attrs["stage_vars"]
        if var in ds.data_vars
    }

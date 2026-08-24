# -*- coding: utf-8 -*-
"""Manual spectrum-exclusion mask: session state, index parsing, digest.

The mask is a plain boolean array over the *spatial* dims only (True =
excluded): ``(n_row, n_col)`` for a map, ``(n_point,)`` for a line scan. Flat
index is C-order ``i = r * n_col + c``, matching every other flatten in the
codebase (``_shared._factorization._flatten_to_row_stack``,
``peak_fitter._batch``'s traversal, ``_shared.scan_geometry``'s meshgrid ravel)
— so a flat index from the selection chart, from a typed spec and from a
downstream row-stack all mean the same spectrum.

Masks live in ``st.session_state["sl_excluded"]`` keyed by filename and are
handed to the pipeline through ``pipeline_params["excl"]["masks"]``. Being part
of the params dict means every existing ``@st.cache_data`` that already takes
``pipeline_params`` as a key invalidates correctly with no extra plumbing.

Everything the *user* sees is 1-based (:data:`DISPLAY_BASE`) — the typed
index fields, the chart hovers and the browse labels — while every array,
mask and flat index stays 0-based. The conversion happens only in
:func:`parse_index_spec` / :func:`parse_pixel_spec` (in) and
:func:`to_display` / :func:`display_range` (out).

Everything here except :func:`get_mask` / :func:`set_mask` / :func:`undo` /
:func:`build_excl_params` is pure — no Streamlit, no session state — so the
parsing and mask algebra can be exercised directly.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

import numpy as np
import streamlit as st
import xarray as xr

_STATE_KEY = "sl_excluded"
_UNDO_KEY = "sl_excluded_undo"
_UNDO_DEPTH = 20

#: Index of the first spectrum/row/column *as the user sees it*. Masks, flat
#: indices and every array operation below stay 0-based — the offset is applied
#: at the two edges of the module (parsing in, labels out) and nowhere else.
DISPLAY_BASE = 1


def to_display(i: int) -> int:
    """0-based internal index → the number shown in the UI."""
    return int(i) + DISPLAY_BASE


def display_range(n: int) -> str:
    """Readable ``"1\u20135"`` inclusive range for ``n`` items, in display numbering."""
    return f"{DISPLAY_BASE}\u2013{n - 1 + DISPLAY_BASE}"


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def spatial_shape(da: xr.DataArray, spectral_dim: str | None = None) -> tuple[int, ...]:
    """Spatial shape of ``da`` — everything except the spectral axis."""
    sdim = spectral_dim if spectral_dim in (da.dims or ()) else da.dims[-1]
    return tuple(int(da.sizes[d]) for d in da.dims if d != sdim)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_PIXEL_RE = re.compile(r"\(?\s*(\d+)\s*[,;]\s*(\d+)\s*\)?")


def parse_index_spec(text: str, n_max: int) -> list[int]:
    """Parse ``"1-4, 8, 11-13"`` into 0-based ``[0,1,2,3,7,10,11,12]``.

    The text is written by the user, so it is in display numbering
    (:data:`DISPLAY_BASE`); the returned indices are the 0-based ones the mask
    is addressed with. Accepts commas and/or whitespace as separators and
    inclusive ``a-b`` ranges (in either order). Raises ``ValueError`` with a
    readable message on malformed tokens or indices outside the valid range.
    """
    if not text or not text.strip():
        return []

    out: set[int] = set()
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        m = _RANGE_RE.match(token)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            values: Iterable[int] = range(lo, hi + 1)
        elif token.isdigit():
            values = (int(token),)
        else:
            raise ValueError(
                f"Could not read {token!r} — use indices and ranges, e.g. '1-4, 8, 11-13'."
            )
        for v in values:
            if not DISPLAY_BASE <= v < n_max + DISPLAY_BASE:
                raise ValueError(
                    f"Index {v} is out of range (valid: {display_range(n_max)})."
                )
            out.add(v - DISPLAY_BASE)
    return sorted(out)


def parse_pixel_spec(text: str, n_row: int, n_col: int) -> list[tuple[int, int]]:
    """Parse ``"(5,8), (10,3)"`` into 0-based ``[(4, 7), (9, 2)]`` — (row, column).

    Like :func:`parse_index_spec`, the text is in display numbering and the
    returned pairs are 0-based. Parentheses are optional; ``5,8 10,3`` parses
    the same. Raises ``ValueError`` on out-of-range coordinates or leftover
    unparsable text.
    """
    if not text or not text.strip():
        return []

    out: list[tuple[int, int]] = []
    consumed = 0
    for m in _PIXEL_RE.finditer(text):
        r, c = int(m.group(1)), int(m.group(2))
        if not DISPLAY_BASE <= r < n_row + DISPLAY_BASE:
            raise ValueError(f"Row {r} is out of range (valid: {display_range(n_row)}).")
        if not DISPLAY_BASE <= c < n_col + DISPLAY_BASE:
            raise ValueError(f"Column {c} is out of range (valid: {display_range(n_col)}).")
        out.append((r - DISPLAY_BASE, c - DISPLAY_BASE))
        consumed += len(m.group(0))

    if not out:
        raise ValueError(
            f"Could not read {text.strip()!r} — use (row, column) pairs, e.g. '(5,8), (10,3)'."
        )
    # Anything that is not a separator between matches is a typo worth flagging.
    leftover = re.sub(r"[(),;\s\d]", "", text)
    if leftover:
        raise ValueError(
            f"Unexpected characters {leftover!r} — use (row, column) pairs, e.g. '(5,8), (10,3)'."
        )
    return out


# ---------------------------------------------------------------------------
# Mask algebra (pure)
# ---------------------------------------------------------------------------

def apply_selection(
    mask: np.ndarray,
    *,
    rows: Sequence[int] = (),
    cols: Sequence[int] = (),
    pixels: Sequence[tuple[int, int]] = (),
    flat: Sequence[int] = (),
    exclude: bool = True,
) -> np.ndarray:
    """Return a new mask with the given selection set to ``exclude``.

    ``rows``/``cols`` are whole map rows/columns and are ignored for 1-D
    (line-scan) masks, where ``flat`` is the natural addressing. Both
    directions are idempotent, which is what makes the chart's
    fires-only-on-change selection callback safe to re-run.
    """
    out = np.array(mask, dtype=bool, copy=True)

    if out.ndim == 2:
        for r in rows:
            out[r, :] = exclude
        for c in cols:
            out[:, c] = exclude
        for r, c in pixels:
            out[r, c] = exclude

    if len(flat):
        flat_idx = np.asarray(flat, dtype=int)
        out.reshape(-1)[flat_idx] = exclude

    return out


def mask_digest(mask: np.ndarray | None) -> str:
    """Short deterministic tag for a mask; ``""`` when nothing is excluded.

    Empty tag means "no exclusion", so an untouched file keeps exactly the
    memo key it had before this feature existed and costs nothing.
    """
    if mask is None:
        return ""
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return ""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(arr.shape).encode())
    h.update(np.packbits(arr.ravel()).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def get_mask(fname: str, shape: tuple[int, ...]) -> np.ndarray:
    """Stored mask for ``fname``, or a fresh all-False one.

    A stored mask whose shape no longer matches is discarded rather than
    reused — re-uploading a same-named file with different scan geometry must
    degrade to "nothing excluded", never to a wrong-shaped mask.
    """
    stored = st.session_state.get(_STATE_KEY, {}).get(fname)
    if stored is not None and tuple(stored.shape) == tuple(shape):
        return stored
    return np.zeros(shape, dtype=bool)


def set_mask(fname: str, mask: np.ndarray) -> None:
    """Store ``mask`` for ``fname``, pushing the previous value onto the undo stack."""
    store = st.session_state.setdefault(_STATE_KEY, {})
    undo: list[tuple[str, np.ndarray | None]] = st.session_state.setdefault(_UNDO_KEY, [])
    undo.append((fname, store.get(fname)))
    del undo[:-_UNDO_DEPTH]
    store[fname] = np.asarray(mask, dtype=bool)


def undo() -> bool:
    """Revert the most recent :func:`set_mask`. Returns False if nothing to undo."""
    undo_stack: list[tuple[str, np.ndarray | None]] = st.session_state.get(_UNDO_KEY) or []
    if not undo_stack:
        return False
    fname, previous = undo_stack.pop()
    store = st.session_state.setdefault(_STATE_KEY, {})
    if previous is None:
        store.pop(fname, None)
    else:
        store[fname] = previous
    return True


def clear_mask(fname: str, shape: tuple[int, ...]) -> None:
    """Restore every spectrum of ``fname`` (undoable)."""
    set_mask(fname, np.zeros(shape, dtype=bool))


def has_undo() -> bool:
    return bool(st.session_state.get(_UNDO_KEY))


def build_excl_params(loaded: dict[str, Any]) -> dict[str, Any]:
    """Assemble the ``pipeline_params["excl"]`` subdict for the loaded files.

    Only non-empty masks are included, so files with nothing excluded keep the
    exact params digest they had before and stay memo hits.
    """
    store = st.session_state.get(_STATE_KEY) or {}
    masks = {
        name: mask
        for name, mask in store.items()
        if name in loaded and mask is not None and np.any(mask)
    }
    return {"masks": masks} if masks else {}

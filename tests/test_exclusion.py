"""1-based display parsing and mask digest — the cache-key contract for exclusion."""

from __future__ import annotations

import numpy as np
import pytest

from frontend.exclusion import (
    DISPLAY_BASE,
    apply_selection,
    mask_digest,
    parse_index_spec,
    parse_pixel_spec,
    to_display,
)


def test_display_base_is_one():
    assert DISPLAY_BASE == 1
    assert to_display(0) == 1
    assert to_display(7) == 8


def test_parse_index_spec_is_one_based():
    assert parse_index_spec("1-4, 8", 20) == [0, 1, 2, 3, 7]
    assert parse_index_spec("20", 20) == [19]
    assert parse_index_spec("", 20) == []
    assert parse_index_spec("  3  ", 20) == [2]


def test_parse_index_spec_rejects_zero_and_overflow():
    with pytest.raises(ValueError, match="out of range"):
        parse_index_spec("0", 20)
    with pytest.raises(ValueError, match="out of range"):
        parse_index_spec("21", 20)
    with pytest.raises(ValueError, match="Could not read"):
        parse_index_spec("abc", 20)


def test_parse_pixel_spec_is_one_based():
    assert parse_pixel_spec("(1,1), (4,5)", 10, 10) == [(0, 0), (3, 4)]
    assert parse_pixel_spec("5,8 10,3", 12, 12) == [(4, 7), (9, 2)]


def test_mask_digest_empty_is_blank_tag():
    assert mask_digest(None) == ""
    assert mask_digest(np.zeros((3, 4), dtype=bool)) == ""


def test_mask_digest_stable_and_sensitive():
    a = np.zeros((3, 4), dtype=bool)
    a[1, 2] = True
    b = a.copy()
    assert mask_digest(a) == mask_digest(b)
    c = a.copy()
    c[0, 0] = True
    assert mask_digest(a) != mask_digest(c)
    d = np.zeros((4, 3), dtype=bool)
    d[1, 2] = True
    assert mask_digest(a) != mask_digest(d)


def test_apply_selection_rows_and_pixels():
    mask = np.zeros((4, 5), dtype=bool)
    out = apply_selection(mask, rows=[1], pixels=[(3, 2)], exclude=True)
    assert out[1, :].all()
    assert out[3, 2]
    assert not out[0, 0]
    np.testing.assert_array_equal(mask, np.zeros((4, 5), dtype=bool))

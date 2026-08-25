"""Sidebar upload identity reuse."""

from __future__ import annotations

from frontend.sidebar import _upload_matches_loaded
from tests.factories import gaussian_map, make_map


class _FakeUpload:
    def __init__(self, name: str, file_id: str):
        self.name = name
        self.file_id = file_id


def test_upload_matches_loaded_same_ids():
    ds = make_map(gaussian_map())
    loaded = {"a.wdf": {"hash": "id1", "dataset": ds}}
    uploads = [_FakeUpload("a.wdf", "id1")]
    assert _upload_matches_loaded(uploads, loaded)


def test_upload_matches_loaded_rejects_new_id():
    ds = make_map(gaussian_map())
    loaded = {"a.wdf": {"hash": "id1", "dataset": ds}}
    uploads = [_FakeUpload("a.wdf", "id2")]
    assert not _upload_matches_loaded(uploads, loaded)


def test_upload_matches_loaded_rejects_extra_file():
    ds = make_map(gaussian_map())
    loaded = {"a.wdf": {"hash": "id1", "dataset": ds}}
    uploads = [_FakeUpload("a.wdf", "id1"), _FakeUpload("b.wdf", "id2")]
    assert not _upload_matches_loaded(uploads, loaded)

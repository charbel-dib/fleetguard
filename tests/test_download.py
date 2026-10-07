import io
import zipfile

import pandas as pd
import pytest

import fleetguard.download as download_module
from fleetguard.data import DESCRIPTION_NAME, TEST_NAME, TRAIN_NAME
from fleetguard.download import _unpack, verify_raw
from fleetguard.io import sha256_file, write_json


def test_allowlisted_extraction_does_not_extract_arbitrary_paths(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in (TRAIN_NAME, TEST_NAME, DESCRIPTION_NAME):
            zipped.writestr("nested/" + name, "fixture")
        zipped.writestr("../outside.txt", "do not extract")
    destination = tmp_path / "raw"
    destination.mkdir()
    _unpack(archive, destination)
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        [TRAIN_NAME, TEST_NAME, DESCRIPTION_NAME]
    )
    assert not (tmp_path / "outside.txt").exists()


def test_modified_source_file_is_detected(tmp_path):
    names = (TRAIN_NAME, TEST_NAME, DESCRIPTION_NAME)
    for name in names:
        (tmp_path / name).write_text("original")
    manifest = {"files": {name: sha256_file(tmp_path / name) for name in names}}
    write_json(tmp_path / "manifest.json", manifest)
    assert verify_raw(tmp_path) == manifest
    (tmp_path / TRAIN_NAME).write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        verify_raw(tmp_path)


def test_network_download_registers_hashes_and_second_call_is_offline(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in (TRAIN_NAME, TEST_NAME, DESCRIPTION_NAME):
            zipped.writestr(name, "fixture")
    archive_bytes = archive.getvalue()
    calls = []

    def open_fixture(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(archive_bytes)

    # Structural APS validation is covered by loader tests and the real benchmark.
    monkeypatch.setattr(download_module.urllib.request, "urlopen", open_fixture)
    monkeypatch.setattr(download_module, "load_aps", lambda _: (pd.DataFrame({"a": [1]}), None))
    destination = tmp_path / "raw"
    manifest = download_module.download(destination)
    assert len(calls) == 1
    assert manifest["source_url"] == download_module.SOURCE_URL
    assert verify_raw(destination) == manifest
    assert download_module.download(destination) == manifest
    assert len(calls) == 1

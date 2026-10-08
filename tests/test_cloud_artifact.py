import stat
import zipfile

import pytest

from fleetguard.cloud.artifact import pack_model, unpack_model
from fleetguard.io import sha256_file
from fleetguard.release_smoke import create_fixture


@pytest.fixture(scope="module")
def model_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("cloud-archive")
    _, _, release = create_fixture(root)
    archive = root / "model.zip"
    receipt = pack_model(release, archive, allow_synthetic=True)
    return release, archive, receipt


def test_archive_roundtrip_is_minimal_deterministic_and_preserves_freeze(model_archive, tmp_path):
    release, archive, receipt = model_archive
    pack_model(release, tmp_path / "same.zip", allow_synthetic=True)
    assert sha256_file(tmp_path / "same.zip") == receipt["bundle_sha256"]
    identity = unpack_model(
        archive, receipt["bundle_sha256"], tmp_path / "bundle", allow_synthetic=True
    )
    assert identity == receipt["model"]
    assert sha256_file(tmp_path / "bundle/freeze.json") == identity["freeze_sha256"]
    with zipfile.ZipFile(archive) as stream:
        assert len(stream.namelist()) == 12
        assert not any("final_test" in name or "predictions" in name for name in stream.namelist())
    with pytest.raises(ValueError, match="already exists"):
        pack_model(release, archive, allow_synthetic=True)


def test_wrong_digest_or_synthetic_archive_cannot_enter_cloud(model_archive, tmp_path):
    _, archive, receipt = model_archive
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        unpack_model(archive, "0" * 64, tmp_path / "bad")
    with pytest.raises(ValueError, match="Synthetic"):
        unpack_model(archive, receipt["bundle_sha256"], tmp_path / "synthetic")
    assert not (tmp_path / "synthetic").exists()


@pytest.mark.parametrize("attack", ["traversal", "symlink", "duplicate", "extra", "corruption"])
def test_archive_rejects_unsafe_members_before_model_load(model_archive, tmp_path, attack):
    _, archive, _ = model_archive
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(damaged, "w") as target:
        for i, member in enumerate(source.infolist()):
            content = source.read(member)
            if i == 0 and attack == "traversal":
                member.filename = "../escape"
            if i == 0 and attack == "symlink":
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            if member.filename.endswith("pipeline.joblib") and attack == "corruption":
                content = b"must never be deserialized"
            target.writestr(member, content)
        if attack == "duplicate":
            with pytest.warns(UserWarning):
                target.writestr("run.json", source.read("run.json"))
        if attack == "extra":
            target.writestr("private.txt", "extra")
    with pytest.raises(ValueError):
        unpack_model(damaged, sha256_file(damaged), tmp_path / "out", allow_synthetic=True)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape").exists()

import pytest

from fleetguard.serving.deployment import deployment_identity


def test_local_provenance_is_explicitly_unknown(monkeypatch):
    for key in ("GIT_SHA", "BUNDLE_SHA256", "IMAGE_DIGEST"):
        monkeypatch.delenv("FLEETGUARD_" + key, raising=False)
    assert deployment_identity() == {"git_sha": None, "bundle_sha256": None, "image_digest": None}


def test_cloud_provenance_is_exact_and_rejects_tags(monkeypatch):
    monkeypatch.setenv("FLEETGUARD_GIT_SHA", "a" * 40)
    monkeypatch.setenv("FLEETGUARD_BUNDLE_SHA256", "b" * 64)
    monkeypatch.setenv("FLEETGUARD_IMAGE_DIGEST", "sha256:" + "c" * 64)
    assert deployment_identity()["image_digest"] == "sha256:" + "c" * 64
    monkeypatch.setenv("FLEETGUARD_IMAGE_DIGEST", "latest")
    with pytest.raises(ValueError, match="image_digest"):
        deployment_identity()

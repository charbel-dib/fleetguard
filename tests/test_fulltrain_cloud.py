import pytest

from fleetguard.artifacts import save_model
from fleetguard.cloud.artifact import pack_model, unpack_model
from fleetguard.fulltrain import COSTS, SeedEnsemble
from fleetguard.io import sha256_file, write_json


@pytest.fixture
def release(classification_frame, tmp_path):
    features, target = classification_frame
    model = SeedEnsemble(seeds=(0,), max_iter=10).fit(features, target)
    directory = tmp_path / "release-cloud"
    directory.mkdir()
    save_model(
        directory / "model",
        model,
        metadata={
            "name": "hgb_seed_ensemble",
            "feature_names": list(features.columns),
            "threshold": 0.5,
            "costs": COSTS,
            "calibration": "none",
            "score_semantics": "ranking score",
            "release_id": directory.name,
        },
    )
    write_json(
        directory / "decision.json",
        {
            "protocol": "fulltrain_v1",
            "dataset_kind": "synthetic_smoke",
            "release_id": directory.name,
            "threshold": 0.5,
            "official_test_used_for_selection": False,
            "model_sha256": sha256_file(directory / "model" / "pipeline.joblib"),
            "config": {"costs": COSTS},
        },
    )
    return directory


def test_pack_and_unpack_roundtrip(release, tmp_path):
    receipt = pack_model(release, tmp_path / "model.zip", allow_synthetic=True)
    assert sorted(receipt["files"]) == [
        "decision.json",
        "model/metadata.json",
        "model/pipeline.joblib",
    ]
    identity = unpack_model(
        tmp_path / "model.zip", receipt["bundle_sha256"], tmp_path / "out", allow_synthetic=True
    )
    assert identity == receipt["model"]


def test_unpack_rejects_wrong_digest(release, tmp_path):
    pack_model(release, tmp_path / "model.zip", allow_synthetic=True)
    with pytest.raises(ValueError, match="SHA256"):
        unpack_model(tmp_path / "model.zip", "0" * 64, tmp_path / "out", allow_synthetic=True)

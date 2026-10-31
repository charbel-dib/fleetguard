import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fleetguard.artifacts import save_model  # noqa: E402
from fleetguard.fulltrain import COSTS, SeedEnsemble, cost_threshold, verify_release  # noqa: E402
from fleetguard.io import sha256_file, write_json  # noqa: E402
from fleetguard.serving.api import create_app  # noqa: E402
from fleetguard.serving.bundle import bundle_release  # noqa: E402
from fleetguard.serving.settings import Settings  # noqa: E402


@pytest.fixture
def release(classification_frame, tmp_path):
    features, target = classification_frame
    model = SeedEnsemble(seeds=(0, 1), max_iter=20).fit(features, target)
    scores = model.predict_proba(features)[:, 1]
    threshold, _ = cost_threshold(target, scores)
    directory = tmp_path / "release-test"
    directory.mkdir()
    save_model(
        directory / "model",
        model,
        metadata={
            "name": "hgb_seed_ensemble",
            "feature_names": list(features.columns),
            "threshold": threshold,
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
            "threshold": threshold,
            "official_test_used_for_selection": False,
            "model_sha256": sha256_file(directory / "model" / "pipeline.joblib"),
            "config": {"costs": COSTS},
        },
    )
    return directory, features, model, threshold


def settings_for(directory, synthetic=True):
    return Settings(release_dir=directory, max_batch_rows=4, allow_synthetic=synthetic)


def rows_of(features, n=3):
    head = features.iloc[:n]
    return head.astype(object).where(head.notna(), None).to_dict(orient="records")


def test_serves_fulltrain_release_with_identical_scores(release):
    directory, features, model, threshold = release
    expected = model.predict_proba(features.iloc[:3])[:, 1]
    with TestClient(create_app(settings_for(directory))) as client:
        assert client.get("/health/ready").json()["release_id"] == "release-test"
        response = client.post("/v1/predict-batch", json={"rows": rows_of(features)})
        assert response.status_code == 200
        body = response.json()
        np.testing.assert_allclose(
            [p["positive_score"] for p in body["predictions"]], expected, atol=1e-15
        )
        assert [p["predicted_label"] for p in body["predictions"]] == [
            "pos" if s >= threshold else "neg" for s in expected
        ]
        assert body["model"]["threshold"] == threshold
        assert body["model"]["freeze_sha256"] == sha256_file(directory / "decision.json")


def test_synthetic_release_needs_fixture_mode(release):
    directory, *_ = release
    with pytest.raises(ValueError, match="fixture mode"):
        with TestClient(create_app(settings_for(directory, synthetic=False))):
            pass


def test_bundle_copies_only_serving_files_and_detects_tampering(release, tmp_path):
    directory, *_ = release
    bundle = bundle_release(directory, tmp_path / "bundle")
    files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    assert files == {"decision.json", "model/pipeline.joblib", "model/metadata.json"}
    verify_release(bundle)
    (bundle / "model" / "pipeline.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        with TestClient(create_app(settings_for(bundle))):
            pass

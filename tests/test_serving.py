import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fleetguard.artifacts import load_model, positive_scores  # noqa: E402
from fleetguard.io import read_json, sha256_file, write_json  # noqa: E402
from fleetguard.release_smoke import create_fixture  # noqa: E402
from fleetguard.serving.api import create_app  # noqa: E402
from fleetguard.serving.bundle import bundle_release  # noqa: E402
from fleetguard.serving.middleware import RequestEnvelope  # noqa: E402
from fleetguard.serving.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def fixture_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-fixture")
    x, _, run = create_fixture(root)
    return x, run


@pytest.fixture
def settings(fixture_release):
    _, run = fixture_release
    return Settings(
        release_dir=run, max_batch_rows=4, max_request_bytes=32768, allow_synthetic=True
    )


@pytest.fixture
def rows(fixture_release):
    x, _ = fixture_release
    return x.iloc[:3].astype(object).where(x.iloc[:3].notna(), None).to_dict(orient="records")


def test_single_batch_cli_parity_identity_and_order(settings, rows, fixture_release, monkeypatch):
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    import fleetguard.serving.predictor as predictor_module

    def no_fit(*args, **kwargs):
        raise AssertionError("Serving must never fit a model/calibrator.")

    monkeypatch.setattr(XGBClassifier, "fit", no_fit)
    monkeypatch.setattr(CalibratedClassifierCV, "fit", no_fit)
    loaded = []
    original = predictor_module.load_model

    def load_once(path):
        loaded.append(path)
        return original(path)

    monkeypatch.setattr(predictor_module, "load_model", load_once)
    _, run = fixture_release
    before = sha256_file(run / "freeze.json")
    x, _ = fixture_release
    model, metadata = load_model(run / "xgboost_release")
    expected = positive_scores(model, x.iloc[:3])
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["status"] == "ready"
        info = client.get("/v1/model").json()
        assert info["feature_names"] == metadata["feature_names"]
        reordered = [dict(reversed(list(row.items()))) for row in rows]
        response = client.post("/v1/predict-batch", json={"rows": reordered})
        assert response.status_code == 200
        batch = response.json()
        np.testing.assert_allclose(
            [p["positive_score"] for p in batch["predictions"]], expected, rtol=0, atol=1e-15
        )
        assert [p["row_index"] for p in batch["predictions"]] == [0, 1, 2]
        assert [p["predicted_label"] for p in batch["predictions"]] == [
            "pos" if p >= metadata["threshold"] else "neg" for p in expected
        ]
        single = client.post("/v1/predict", json={"sensors": rows[0]})
        assert single.json()["prediction"] == batch["predictions"][0]
        assert batch["model"] == single.json()["model"] == info["model"]
        assert batch["model"]["freeze_sha256"] == before
        assert response.headers["X-Request-ID"] != single.headers["X-Request-ID"]
        assert len(response.headers["X-Request-ID"]) == 32
        openapi = client.get("/openapi.json").json()
        assert (
            openapi["components"]["schemas"]["BatchRequest"]["properties"]["rows"]["maxItems"] == 4
        )
        assert client.get("/docs").status_code == 200
    assert app.state.predictor is None
    assert len(loaded) == 1
    assert sha256_file(run / "freeze.json") == before


@pytest.mark.parametrize("value", [True, "12", {}, [], 1e100])
def test_invalid_numeric_values_are_422_without_input_echo(settings, rows, value):
    payload = dict(rows[0])
    payload[next(iter(payload))] = value
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict", json={"sensors": payload})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert all("input" not in issue for issue in error["details"])
    assert response.headers["X-Request-ID"] == error["request_id"]


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "{", '{"sensors":{"a":1,"a":2}}'])
def test_malformed_or_ambiguous_json_is_400(settings, text):
    body = text if text.startswith("{") else '{"sensors":{"a":' + text + "}}"
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predict", content=body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.parametrize("change", ["missing", "extra", "empty"])
def test_exact_sensor_schema_is_required(settings, rows, change):
    row = dict(rows[0])
    if change == "missing":
        row.pop(next(iter(row)))
    elif change == "extra":
        row["class"] = 0
    else:
        row = {}
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict", json={"sensors": row})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_sensor_schema"


@pytest.mark.parametrize(
    "payload", [{"rows": []}, {"rows": [None]}, {"rows": {}, "threshold": 0.9}]
)
def test_invalid_batch_shapes(settings, payload):
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict-batch", json=payload)
    assert response.status_code == 422


def test_batch_cap_and_client_cannot_override_release_or_threshold(settings, rows):
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/predict-batch", json={"rows": [rows[0]] * 5}).status_code == 422
        for field, value in (("threshold", 0.5), ("release", "another"), ("model", "another")):
            assert (
                client.post("/v1/predict", json={"sensors": rows[0], field: value}).status_code
                == 422
            )
        # Required keys can be explicitly null; omitted keys cannot be inferred.
        all_null = {name: None for name in rows[0]}
        response = client.post("/v1/predict", json={"sensors": all_null})
        assert response.status_code == 200
        assert response.json()["prediction"]["missing_fraction"] == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "text/plain"},
        {"Content-Type": "application/json", "Content-Encoding": "gzip"},
    ],
)
def test_media_type_and_compression_are_rejected(settings, headers):
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict", content=b"{}", headers=headers)
    assert response.status_code == 415


@pytest.mark.parametrize("content_length", [None, b"1"])
def test_streamed_size_limit_cannot_be_bypassed_by_header(content_length):
    invoked = []
    messages = [
        {"type": "http.request", "body": b" " * 32, "more_body": True},
        {"type": "http.request", "body": b" " * 32, "more_body": True},
        {"type": "http.request", "body": b" ", "more_body": False},
    ]
    sent = []

    async def app(scope, receive, send):
        invoked.append(True)

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    headers = [(b"content-type", b"application/json")]
    if content_length:
        headers.append((b"content-length", content_length))
    asyncio.run(
        RequestEnvelope(app, max_bytes=64)(
            {"type": "http", "method": "POST", "path": "/v1/predict", "headers": headers},
            receive,
            send,
        )
    )
    assert not invoked
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "body_too_large"


def test_early_declared_body_limit_and_invalid_header(settings):
    with TestClient(create_app(settings)) as client:
        oversized = client.post(
            "/v1/predict",
            content=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "32769"},
        )
        assert oversized.status_code == 413
        invalid = client.post(
            "/v1/predict",
            content=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "-1"},
        )
        assert invalid.status_code == 400


def test_liveness_does_not_claim_model_readiness_before_startup(settings, rows):
    app = create_app(settings)
    with TestClient(app, backend="asyncio") as client:
        assert client.get("/health/ready").status_code == 200
    client = TestClient(app)
    try:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/v1/model").status_code == 503
        assert client.post("/v1/predict", json={"sensors": rows[0]}).status_code == 503
    finally:
        client.close()


def test_failed_startup_rejects_tamper_partial_and_unapproved_fixture(settings, tmp_path):
    with pytest.raises(ValueError, match="fixture mode"):
        with TestClient(create_app(replace(settings, allow_synthetic=False))):
            pass
    copy = bundle_release(settings.release_dir, tmp_path / "bundle")
    (copy / "xgboost_release/pipeline.joblib").write_bytes(b"untrusted corrupted model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        with TestClient(create_app(replace(settings, release_dir=copy))):
            pass
    other = bundle_release(settings.release_dir, tmp_path / "partial")
    state = read_json(other / "run.json")
    state["status"] = "running"
    write_json(other / "run.json", state)
    with pytest.raises(ValueError, match="complete training run"):
        with TestClient(create_app(replace(settings, release_dir=other))):
            pass


def test_bundle_contains_only_contract_files_and_is_immutable(settings, tmp_path):
    bundle = bundle_release(settings.release_dir, tmp_path / "bundle")
    expected = set(read_json(settings.release_dir / "freeze.json")["files"]) | {
        "freeze.json",
        "run.json",
    }
    assert {
        str(p.relative_to(bundle)).replace("\\", "/") for p in bundle.rglob("*") if p.is_file()
    } == expected
    assert not (bundle / "validation_predictions.csv").exists()
    with pytest.raises(FileExistsError):
        bundle_release(settings.release_dir, bundle)
    with TestClient(create_app(replace(settings, release_dir=bundle))) as client:
        assert client.get("/health/ready").status_code == 200


def test_internal_error_is_opaque_and_keeps_request_id(settings, rows, monkeypatch):
    app = create_app(settings)
    with TestClient(app) as client:

        def bad_scores(features):
            return np.full((len(features), 2), np.nan)

        monkeypatch.setattr(app.state.predictor._model, "predict_proba", bad_scores)
        response = client.post("/v1/predict", json={"sensors": rows[0]})
        assert response.status_code == 500
        assert response.json()["error"]["message"] == "Prediction failed."
        assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]
        assert str(settings.release_dir) not in response.text


def test_health_is_responsive_while_inference_runs_in_worker_thread(settings, rows, monkeypatch):
    app = create_app(settings)
    entered, release = threading.Event(), threading.Event()
    with TestClient(app) as client:
        original = app.state.predictor._model.predict_proba

        def block(features):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("fixture timed out")
            return original(features)

        monkeypatch.setattr(app.state.predictor._model, "predict_proba", block)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(client.post, "/v1/predict", json={"sensors": rows[0]})
            try:
                assert entered.wait(3)
                assert pool.submit(client.get, "/health/live").result(timeout=2).status_code == 200
            finally:
                release.set()
            assert future.result(timeout=3).status_code == 200


def test_settings_are_explicit_and_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("FLEETGUARD_RELEASE_DIR", raising=False)
    with pytest.raises(ValueError, match="FLEETGUARD_RELEASE_DIR"):
        Settings.from_env()
    with pytest.raises(ValueError, match="max_batch_rows"):
        Settings(release_dir=tmp_path, max_batch_rows=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        Settings(release_dir=tmp_path, max_concurrency=1)
    monkeypatch.setenv("FLEETGUARD_RELEASE_DIR", str(tmp_path))
    monkeypatch.setenv("FLEETGUARD_ALLOW_SYNTHETIC", "not-a-boolean")
    with pytest.raises(ValueError, match="true or false"):
        Settings.from_env()

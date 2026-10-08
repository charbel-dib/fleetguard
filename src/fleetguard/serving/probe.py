"""Standard-library HTTP contract probe for a local API or its Docker container."""

import json
import time
import urllib.error
import urllib.request

import numpy as np


def request_json(url, *, payload=None, timeout=5):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body is not None else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return (
            response.status,
            json.load(response),
            {k.lower(): v for k, v in response.headers.items()},
        )


def probe(url, *, expected_predictor=None, rows=None, expect_synthetic=False, wait_seconds=30):
    url = url.rstrip("/")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, ready, _ = request_json(url + "/health/ready")
            if status == 200 and ready["status"] == "ready":
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("API did not become ready within the probe deadline.")
        time.sleep(0.1)
    status, live, _ = request_json(url + "/health/live")
    if status != 200 or live["status"] != "alive":
        raise RuntimeError("Liveness failed.")
    _, info, _ = request_json(url + "/v1/model")
    kind = "synthetic_smoke" if expect_synthetic else "official_aps"
    if info["model"]["dataset_kind"] != kind:
        raise RuntimeError("The API serves the wrong dataset kind.")
    if rows is None:
        row = {name: None for name in info["feature_names"]}
        rows = [row, {name: 0.0 for name in row}, row][: info["limits"]["max_batch_rows"]]
    _, batch, headers = request_json(url + "/v1/predict-batch", payload={"rows": rows})
    _, single, _ = request_json(url + "/v1/predict", payload={"sensors": rows[0]})
    if batch["rows"] != len(rows) or len(batch["predictions"]) != len(rows):
        raise RuntimeError("Batch row count differs.")
    if batch["predictions"][0] != single["prediction"]:
        raise RuntimeError("Single/batch scores differ.")
    if batch["model"] != info["model"] or single["model"] != info["model"]:
        raise RuntimeError("Model identity changed between API responses.")
    if not headers.get("x-request-id"):
        raise RuntimeError("Request identity missing.")
    scores_match = None
    if expected_predictor is not None:
        expected = expected_predictor.predict(rows)
        np.testing.assert_allclose(
            [row["positive_score"] for row in batch["predictions"]],
            [row["positive_score"] for row in expected],
            rtol=0,
            atol=1e-15,
        )
        if [row["predicted_label"] for row in batch["predictions"]] != [
            row["predicted_label"] for row in expected
        ]:
            raise RuntimeError("API labels differ from local artifact.")
        if batch["model"] != expected_predictor.identity:
            raise RuntimeError("API identity differs from the expected release.")
        scores_match = True
    return {
        "service_version": info["service_version"],
        "contract_version": info["contract_version"],
        "model": info["model"],
        "features": len(info["feature_names"]),
        "tested_rows": len(rows),
        "health_live": "passed",
        "health_ready": "passed",
        "single_batch_parity": True,
        "local_score_parity": scores_match,
        "limits": info["limits"],
        "scope": "HTTP contract/parity check; not a capacity or latency benchmark",
    }

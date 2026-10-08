# Inference API — contract v1, service 0.5.0

One trusted, complete `release_v1` is loaded during startup. The freeze, split and model checks
run before deserialization; a warmup prediction must succeed before readiness. The pipeline,
calibration and threshold then remain in memory until shutdown. No endpoint trains, recalibrates,
changes the policy, uploads models or re-evaluates official test labels.

## Routes

| Method | Path | Result |
|---|---|---|
| GET | `/health/live` | 200: application responds, with service version |
| GET | `/health/ready` | 200: verified model available; otherwise 503 |
| GET | `/v1/model` | Model identity, ordered features, costs and input limits |
| POST | `/v1/predict` | One sensor row and its prediction |
| POST | `/v1/predict-batch` | Ordered sensor rows and their predictions |
| GET | `/openapi.json` | Generated OpenAPI schema |
| GET | `/docs` | Interactive API documentation |

Run `python -m fleetguard serve --release <trusted-release>` with the `serve` extra installed.
The CLI defaults to `127.0.0.1:8000`, one worker and no access log. Model failures abort startup;
the CLI does not silently substitute a baseline. `create_app(Settings(...))` is the explicit ASGI
factory for tests/embedding. Uvicorn must run the lifespan protocol.

## Inputs

Single: `{"sensors": {"aa_000": 12, "ab_000": null, "...": "all remaining sensor keys"}}`.
Batch: `{"rows": [{"aa_000": 12, "ab_000": null, "...": "all remaining sensor keys"}]}`.
These illustrate structure only: the ellipsis is **not** a valid sensor or value.
Use `/v1/model` or `scripts/make-api-request.py` to obtain the real schema and a valid payload.

- Exactly the frozen feature keys are required on every row: 170 for an official APS release.
  Their JSON order may differ. Labels, row IDs and unknown columns are rejected.
- Values are JSON numbers or explicit `null`. Numeric strings, booleans, lists and objects are
  rejected. A missing key differs from an explicitly missing measurement.
- Numbers must be finite and within float32 range (absolute value <= 3.4028234663852886e38),
  because the frozen XGBoost inference path uses float32. `NaN`, `Infinity` and duplicate keys
  are invalid JSON for this contract. An out-of-range exponent is also rejected.
- No caller-supplied threshold, policy or model override. Top-level extra fields are forbidden.
- `Content-Type: application/json` is required for prediction POSTs. Compressed bodies are refused.
- An all-null row exercises the contract, but does not establish a useful truck diagnosis.

The body limit counts actual streamed bytes, including chunked requests; an absent or understated
Content-Length cannot bypass it. A body that exceeds the cap is rejected before parsing/inference.
Pydantic validates values and batch length; the predictor checks feature membership and restores order.

## Outputs and errors

Each successful prediction has `row_index`, `positive_score`, `predicted_label` and `missing_fraction`.
`row_index` is the zero-based position **inside this request**, not a dataset or vehicle identifier.
Single requests return index 0. Batch order is preserved. `pos` means
`positive_score >= model.threshold`; `neg` describes failures outside APS, not healthy vehicles.
The score uses the saved calibration map; reliability on another fleet is not established.

Both prediction responses include `model`: `release_id`, model `name`, `dataset_kind`, pipeline and
freeze SHA-256, threshold, calibration, policy and score semantics. `/v1/model` also exposes
`service_version`, `contract_version: 1`, costs, features, limits and label meanings.
Clients can compare identity across requests and reject an unexpected promotion later.

Application errors use:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Request does not satisfy the input contract.",
    "details": [],
    "request_id": "server-generated-identifier"
  }
}
```

| Status | Meaning |
|---:|---|
| 400 | Malformed/ambiguous JSON or invalid Content-Length |
| 413 | Declared or received body too large |
| 415 | Unsupported content type or content encoding |
| 422 | Wrong schema, missing/extra features, invalid values or batch length |
| 500 | Inference failed; no internal exception/path/body is returned |
| 503 | Model unavailable; readiness uses `{"status":"not_ready"}` |

Application responses carry a newly generated `X-Request-ID`; validation details omit sensor values.
Unexpected failures log the request ID and exception type without a stack trace or input body.
Uvicorn can itself reject excess concurrency with 503 **before** the application middleware;
that response is not the application JSON envelope and has no guaranteed request ID.

## Resource settings

| Environment variable | Default | Accepted range |
|---|---:|---|
| `FLEETGUARD_RELEASE_DIR` | Required unless `--release` is passed | Trusted local release/bundle path |
| `FLEETGUARD_MAX_BATCH_ROWS` | 256 | Integer 1–1024 |
| `FLEETGUARD_MAX_REQUEST_BYTES` | 2097152 (2 MiB) | Integer 1–16777216 |
| `FLEETGUARD_MAX_CONCURRENCY` | 16 | Integer 2–64, CLI Uvicorn connection/task cap |
| `FLEETGUARD_ALLOW_SYNTHETIC` | false | true/false; fixtures only, never production evidence |

Inference runs through one AnyIO worker slot and a predictor lock, with native thread pools limited
to one during prediction. This keeps admitted health requests responsive while a prediction runs
and bounds simultaneous inference memory. It does not establish load capacity or prevent saturation.
`max_concurrency` is enforced by the CLI server, not by an arbitrary ASGI embedding.

## Minimal bundle and container

`python -m fleetguard bundle --release <source> --output <new-directory>` copies the ten hashed
contract files plus `freeze.json` and `run.json`: twelve files. It excludes raw measurements,
individual predictions, final-test reports and tracking databases. The role manifests preserve
partition row/group identifiers, so the bundle is still a local artifact, not a public download.
The destination must be new; failed preparation removes only that new destination.

The Dockerfile installs locked core + `serve` dependencies as a non-editable package, without
PyTorch/Optuna/MLflow. The image contains no fitted model. Compose mounts a locally prepared bundle
read-only and publishes localhost only, using UID 10001, a read-only root filesystem and `/tmp`
tmpfs. The readiness HTTP route supplies the image healthcheck.

Use only an artifact you trust: joblib can execute code, and these checksums detect accidental
modification, not the authenticity of a malicious replacement. Runtime promotion/restarts and
signed artifact provenance are later deployment work. The startup model is pinned in memory;
files are not rehashed per request. Do not mutate the mounted source while starting the service.

This increment is a local service with no authentication, browser CORS or public hosting.
Framework telemetry is disabled explicitly. Public access control, observability, timeouts/load
measurement and deployment belong to the later cloud/operations increments.

## Verification

The API tests cover startup integrity, exact schemas and numerical bounds, streamed byte limits,
single/batch/local score parity, one model load with forbidden fit calls, opaque errors, shutdown
and liveness while an inference worker is blocked. An additional smoke uses a real loopback HTTP
server. [Reference evidence](results/api-05.json) uses four **training** rows and the unchanged
official frozen release. It is a contract check, not an accuracy or HTTP performance estimate.

CI runs the same checks on Linux/Windows and tests an installed wheel from outside `src`.
A separate job builds/runs the image using an explicitly synthetic offline fixture, compares HTTP
to its mounted artifact, checks non-root execution/read-only mounts, absence of research libraries
and container health. Docker execution was unavailable during local delivery; confirm this job
on GitHub and try your official bundle locally before marking container integration complete.

References: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[Pydantic configuration](https://docs.pydantic.dev/latest/api/config/),
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

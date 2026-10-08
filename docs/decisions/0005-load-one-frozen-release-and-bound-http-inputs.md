# ADR 0005 — load one frozen release and bound HTTP inputs

Status: accepted for increment 05.

The research protocol already selected/calibrated a champion and froze its threshold. Serving
must preserve that decision, expose its identity and reject ambiguous sensor input. Model loading
is relatively expensive and trusted joblib deserialization must not be driven by a client request.

Use a FastAPI factory with an explicit lifespan. Verify one complete `release_v1`, deserialize its
whole pipeline once, warm it up, then mark it ready. Missing, altered or unsupported releases fail
startup. There is no training or model-switch endpoint. Restarting with an explicitly selected
bundle is the only model change supported in this increment.

Require exact feature membership and typed finite float32-compatible numbers/null. Reject duplicate
JSON keys and compressed bodies. Count actual streamed bytes before parsing, and bound batch length.
Report policy/calibration/threshold/hashes with every prediction; keep request-local row order.

Run synchronous scientific inference off the event loop, serialize it with one AnyIO slot and a
predictor lock, and limit native pools during prediction. This protects normal health responsiveness
and avoids racing process-level threadpool controls. Uvicorn separately caps admitted concurrency.
Capacity and overload measurements are future operations work, not inferred from a successful smoke.

Package code/dependencies separately from the model. Docker installs core + serving extras only;
mount a verified minimal bundle read-only. CI builds/runs a tiny synthetic bundle to avoid downloading
Scania data or retraining an official champion on every pull request. Official parity is checked
locally on training sensor rows, without opening official-test labels or changing the freeze.

Consequences: API contracts and artifacts are independently inspectable; model promotion and browser
access policy remain explicit later tasks. Checksums are not signatures, and trusted artifacts remain
a deployment requirement. A model pin plus restart is adequate locally; hot reload would add lifecycle
and reproducibility risks before a promotion/rollback protocol exists.

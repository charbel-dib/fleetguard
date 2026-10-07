# FleetGuard — implementation sequence

The complete status checklist is [PROJECT_CHECKLIST.md](../PROJECT_CHECKLIST.md).
This sequence describes code and acceptance criteria, not a time estimate.

## 1. Data and baseline foundation — delivered

Source import, data validation, grouped holdout, trivial/logistic baselines, cost-aware thresholds,
pipeline persistence, batch inference and CI. The first push is confirmed.

## 2. Train-only comparison — delivered in update 02

Three development CV folds, separate fitting/calibration/threshold roles, Random Forest, HGB,
CPU XGBoost, frozen HGB sigmoid calibration, initial preprocessing ablations, OOF metrics,
error/reliability summaries and reproducible figures. Apply the delta archive and follow UPDATE_02.md
for the feature branch, checks, push and PR. A local implementation does not establish a remote CI result.

## 3. Optimization and ML coverage

Add bounded Optuna search inside development, MLflow experiment tracking, train-fold-only
resampling, deeper preprocessing ablations, multi-seed stability and learning curves.
Build a PyTorch MLP with explicit training/evaluation modes, early stopping, checkpointing and
controlled comparison against tree models. Confirm GPU identity/VRAM before selecting GPU settings.

Acceptance: reproducible search budgets, controlled data roles and evidence that gains persist.

## 4. Interpretation and experimental freeze

Add permutation importance/SHAP where appropriate, detailed FP/FN studies and calibrated-score
assessment. Measure inference latency and memory, define the inspection operating point,
write the model card, then perform the frozen official-test evaluation.

Acceptance: the selected model, threshold and preprocessing are fixed before test evaluation;
reported limits reflect the anonymized historical dataset.

## 5. Inference service

Add serving/predictor.py and FastAPI/Pydantic endpoints: health, readiness, individual prediction
and batch prediction. Validate schema and limits, report model version, compare API/local scores
and package a trusted artifact with Docker.

Acceptance: deterministic inference contract, bad-input handling and API/container tests.

## 6. Professional web application

Build React/TypeScript batch upload and alert review, a diagnostics view and an experiment viewer.
Clearly distinguish labeled validation tradeoffs from unlabeled production input. Add accessibility,
loading/error handling, API integration tests and a production build.

Acceptance: a visitor can use an example, inspect an alert and understand the measured tradeoff.

## 7. Cloud and CI/CD

Declare artifact storage and hosting, build/publish container images, test staging, promote models,
deploy frontend/backend and verify rollback. Integrate infrastructure configuration with versioned
releases rather than committing generated models or secrets.

Acceptance: automated deployment for a real service, recorded model version and working rollback.

## 8. Operations

Add structured logs, service metrics, fixed load benchmarks, error/latency monitoring, input drift
checks, hosting budget and maintenance instructions. Measure the demo's practical limits.

Acceptance: health, capacity and failure behavior are inspectable.

## 9. Portfolio integration

Publish the Vercel portfolio case study with demo, figures, results, architecture, GitHub link,
model/data versions and reproduction steps. Finish the README using measured behavior and
screenshots of the actual application.

Acceptance: a public example works, the project evidence is understandable and every status
claimed as complete has been verified.

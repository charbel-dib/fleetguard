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

## 3. Optimization and ML coverage — delivered in update 03

Bounded sequential Optuna search for XGBoost/MLP, local MLflow SQLite tracking, fit-only random
oversampling, separate signed-log/RobustScaler controls, partition-seed stability and data-size
learning curves. The PyTorch MLP selects epochs on a group-aware holdout inside fit, then refits
fit before separate sigmoid calibration. CPU is the locked default.

Acceptance: budgets and roles are inspectable; experiments, limitations, tracking and portable
inference are verified. Improved CV selection cost is not treated as an independent test gain.
Follow UPDATE_03.md for the branch, Windows environment scripts, checks and PR.

## 4. Interpretation and experimental freeze — delivered in update 04

Grouped calibration-role raw/sigmoid selection, fixed-threshold permutation audits (columns and
prefix groups), FP/FN/missingness slices, frozen inspection scenarios, isolated CPU resource
microbenchmark, verified inference freeze, model card and one-shot official-test evaluation.
The shared local receipt is consumed before labels are loaded and remains consumed after failure.
The benchmark excludes HTTP/network; runtime memory includes the interpreter and libraries.

Acceptance: the selected model, threshold and preprocessing are fixed before test evaluation;
reported limits reflect the anonymized historical dataset. The reference experiment is complete;
follow UPDATE_04.md to reproduce it locally and integrate the feature branch through CI/PR.
Future implementation stages reuse the frozen release, without tuning on published test outcomes.

## 5. Inference service — delivered in update 05; integration pending

FastAPI/Pydantic health/readiness, single/batch prediction, exact frozen schema, streamed body and
row caps, strict finite numeric input, identity hashes and startup-only verified loading. Inference
runs outside the event loop with one worker slot. A twelve-file minimal bundle is mounted read-only
by an unprivileged serving-only Docker image. Windows scripts and CI include the serving extra.

Acceptance: API tests and real HTTP/local parity passed on the frozen reference release; no training,
retuning or official-test re-evaluation. Follow UPDATE_05.md for your release and feature branch.
The image build/run job is provided, but Docker was unavailable during local verification. Confirm
container health/parity locally and green remote Linux/Windows/container CI before merging.
Capacity, browser CORS and public deployment are subsequent work.

## 6. Professional web application — delivered in update 06; integration pending

React/TypeScript strict CSV import, bounded sequential HTTP batches, cancellation and model identity
checks, paginated/filterable review, sensor dialog, session review marks and JSON export. Reference
validation/test diagnostics and CV experiments are visibly separated from active model/unlabeled lot
quantities. The production build can share the local API origin; Vite uses a development proxy.

Acceptance: 28 frontend unit cases, seven real-HTTP browser workflows, keyboard/axe/mobile checks,
104 Python tests and both builds verified under Linux. Follow UPDATE_06.md for your Windows release
and PR. Confirm remote CI, then deploy in stage 7; localhost delivery is not public hosting.

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

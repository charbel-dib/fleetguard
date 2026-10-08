# FleetGuard

Cost-aware classification of Air Pressure System (APS) failures from Scania truck operational data.

An unnecessary inspection and a missed failure do not have the same cost. FleetGuard compares
reference classifiers, regularized logistic regression and tree ensembles, then selects a decision
threshold using the challenge cost: **10 × false positives + 500 × false negatives**.

The package includes verified source snapshots, group-aware cross-validation, isolated calibration
and threshold tuning, saved inference pipelines, budgeted Optuna search, a PyTorch MLP, local MLflow tracking,
experiment figures and CI. Version 0.4 adds a calibrated frozen release, permutation/error audits,
CPU resource measurements, a model card and a one-shot official-test evaluation.
Version 0.5 serves that release through a bounded FastAPI/Pydantic HTTP contract, with a minimal
artifact bundle, a serving-only Dockerfile and container CI. Version 0.6 adds a React/TypeScript CSV review interface, published evidence views and browser tests.

## What is being predicted?

The public dataset contains operational measurements from heavy trucks:

- `pos`: failure associated with a specific APS component.
- `neg`: failure associated with components outside APS. These are **not healthy-truck labels**.
- 60,000 official training rows, including 1,000 positives; 16,000 official test rows.
- 170 anonymized numerical features, plus the `class` column. Missing values are written as `na`.

There is no usable timestamp or truck identifier in the released table. This is a diagnostic
classification benchmark, not evidence of forecasting future breakdowns on unseen fleets.
The original feature names, including `am_0` and `ec_00`, are preserved.

Source: [UCI APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks),
[DOI: 10.24432/C51S51](https://doi.org/10.24432/C51S51).

## Quick start

Use Python **3.11 or 3.12** and run commands from the repository root.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available.
`uv.lock` pins the runtime and development dependencies. `uv sync` creates `.venv` and installs
FleetGuard as an editable package; no `PYTHONPATH` configuration is needed.

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev fleetguard download
uv run --frozen --extra dev fleetguard validate
uv run --frozen --extra dev fleetguard train
```

For the model comparison introduced in version 0.2:

```bash
uv run --frozen --extra dev fleetguard compare
```

Existing users: apply the latest changed files using [UPDATE_06.md](UPDATE_06.md).
Optimization is documented in [UPDATE_03.md](UPDATE_03.md); its Windows smoke repair is in
[HOTFIX_03_01.md](HOTFIX_03_01.md).
The preceding comparison update is documented in [UPDATE_02.md](UPDATE_02.md).

On Windows, use the scripts below if the repository is under OneDrive.
The archive download is about 54 MiB. The complete training
run needs substantially more memory than the download size because parsing and preprocessing
materialize numerical arrays. A CPU is sufficient for this increment.

If the automatic download is blocked, download the ZIP from UCI and import it:

```powershell
uv run --frozen --extra dev fleetguard download --archive "C:\Downloads\aps+failure+at+scania+trucks.zip"
```

Download/import validates the original files and writes `data/raw/manifest.json`. Repeating
`download` verifies the existing snapshot. Files are never silently overwritten. SHA-256 hashes
identify the downloaded snapshot; they are not a signature supplied by UCI.

### Inspect a run

`train` prints the validation comparison and the new run directory. Each run contains:

```text
artifacts/runs/<run-id>/
├── config.json
├── source.json
├── environment.json
├── data_profile.json
├── split.csv
├── run.json
├── champion.json
├── validation_metrics.csv
├── validation_report.md
├── always_negative/
├── class_prior/
├── logistic/
└── logistic_balanced/
```

Each model directory contains `pipeline.joblib`, `metadata.json`, `threshold_curve.csv` and
`validation_predictions.csv`. The metadata records the ordered input schema, threshold,
cost assumptions, validation results and scikit-learn version. `row_id` is the zero-based position
in the parsed source CSV, after its header; it is not a vehicle identifier.

### Batch inference

Provide a plain CSV with exactly the 170 sensor columns and no target. Column order can differ;
it is restored to the training schema. Use `na` or empty fields for missing values.

```powershell
$run = "artifacts/runs/REPLACE_WITH_THE_RUN_ID"
uv run --frozen --extra dev fleetguard predict --run $run --input sensors.csv --output reports/predictions.csv
```

The output contains `row_id`, `positive_score`, `predicted_label`, `threshold` and `model`.
The original UCI files contain a copyright preamble and labels; they are not plain inference CSVs.
An example conversion is documented in [START_HERE.md](START_HERE.md).

### Final test

During development, use validation results. Once model and threshold choices are fixed:

```powershell
uv run --frozen --extra dev fleetguard evaluate-test --run $run --final
```

This evaluates only the validation-selected champion, without retraining or threshold adjustment.
Costs are read from the saved model, not from a potentially edited config. The command checks the
source snapshot and refuses to overwrite an existing final-test report for that run. It cannot
prevent test reuse across different legacy runs; preserving the test holdout remains an experimental rule.
The current audited-release protocol below adds a snapshot-wide local receipt.

## First experiment

| Candidate | Purpose | Preprocessing |
|---|---|---|
| Always negative | Expose the limits of accuracy on an imbalanced dataset | None |
| Class prior | Constant score equal to training prevalence | None |
| Logistic regression | Linear decision boundary with regularization | Median imputation, missing indicators, scaling |
| Balanced logistic regression | Measure the effect of inverse-frequency class weights | Same pipeline |

The official training file is divided with `StratifiedGroupKFold`: five folds, seed 42,
fold 0 held out. Exact duplicate feature rows share a group, even when their labels differ.
The validation size and class ratio are therefore approximate, rather than forced to exactly 20%.
Imputation and scaling are fitted only on the remaining training rows.

Both logistic variants use `C=1.0` and the binary `liblinear` solver. They are reference models;
these settings have not been tuned. A convergence warning fails the run instead of silently
publishing an unconverged model.

For each candidate, the report includes results at threshold 0.5 and the minimum-cost validation
threshold. The always-negative reference keeps its fixed rule. Model selection uses validation
cost, then average precision. Threshold search includes all distinct observed scores and both
extreme decision rules; tied scores are never split. Equal-cost thresholds prefer fewer missed
failures, then fewer false alarms.

Metrics include average precision, precision, recall, F1, ROC-AUC, Brier score, inspection rate,
confusion counts and total cost. Cost is measured in challenge units, not euros. Model scores,
especially from weighted logistic regression, are **not calibrated failure probabilities**.

See [methodology](docs/methodology.md), [learning notes](docs/learning-notes.md), and
[the first design decision](docs/decisions/0001-evaluation-foundation.md).

### Recorded validation run

The initial full-data benchmark was executed on 2026-10-07 with the locked Python 3.12
environment. The holdout contains 12,000 observations, including 200 positives.

| Candidate | Rule | Recall | Precision | FP | FN | Cost |
|---|---|---:|---:|---:|---:|---:|
| Always negative | Fixed | 0.00% | 0.00% | 0 | 200 | 100,000 |
| Logistic | Threshold 0.5 | 64.50% | 86.00% | 21 | 71 | 35,710 |
| Logistic | Selected threshold | 90.50% | 44.58% | 225 | 19 | 11,750 |
| Balanced logistic | Selected threshold | 92.50% | 30.33% | 425 | 15 | 11,750 |

The two logistic variants tie on cost; the unweighted model is selected by its higher average
precision (0.8008 vs 0.7658). Thresholds were selected on this holdout, so these development
results should not be interpreted as unbiased final-test estimates. At the baseline stage, the official test had not
been evaluated; final release results are now recorded below. Full metrics and source/environment provenance are in
[docs/results/baseline-validation.md](docs/results/baseline-validation.md).

## Train-only model comparison

`compare` reads `configs/comparison.toml`. It preserves the original 48,000/12,000 development
split, then compares ten candidates using three scoring folds within development. Each fold's
pool has separate fitting, calibration and threshold roles. The scoring labels never select the
threshold, and retained validation labels never select the champion.

Candidates add Random Forest, native-missing-value HGB, CPU XGBoost, HGB sigmoid calibration,
a missing-indicator ablation and a signed-log/robust-scaling variant. The HGB calibrator wraps
the already-fitted model using `FrozenEstimator`. All candidates use equal fitting sample sizes.
HGB's internal random early-stopping holdout is disabled; budgets are fixed in the config.

The resulting run under `artifacts/comparison/` contains CV metrics, OOF predictions, role
manifests, error summaries, figures, and only the CV-selected champion's final artifact. The
existing `predict --run ...` command accepts that run. Full details are in
[docs/comparison-methodology.md](docs/comparison-methodology.md).

The full Scania run recorded for this increment selected XGBoost with an out-of-fold cost of
38,740 over 48,000 scored observations. Its frozen retained-validation decision produced
181 false positives and 17 false negatives: cost 10,310, recall 91.5%, precision 50.27%.
At that comparison stage, the official test had not been evaluated. These results use a different protocol and fitting
sample size from the original baseline and are not a same-data before/after comparison.
The measured comparison and figures are in [docs/results/comparison-02.md](docs/results/comparison-02.md).

## Budgeted optimization and a PyTorch MLP

Version 0.3 adds the optional `research` extra. Existing baseline/comparison commands work
without it. The locked research environment uses PyTorch CPU on Linux/Windows, independent of CUDA.

```bash
uv sync --frozen --extra dev --extra research
uv run --no-sync python -m fleetguard optimize
```

On Windows, especially for a repository under OneDrive:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1 -Research
. .\scripts\use-environment.ps1 -Research
uv run --no-sync python -m fleetguard optimize
```

The scripts place the environment/cache in `LOCALAPPDATA`, force copy mode, stop after errors,
and use the selected Python rather than an executable from Conda. Configure each new terminal
with `use-environment.ps1`. This avoids the hardlink/partial-installation errors encountered
while upgrading. The checks script runs baseline/comparison/release smokes plus tests, lint and build;
`-Research` adds optimization and `-Serve` adds the real HTTP API smoke.

`configs/optimization.toml` defines six sequential XGBoost trials and four MLP trials. Each uses
three development scoring folds with the same fitting/calibration/threshold roles as comparison.
A group-aware split **inside fit** selects MLP epochs. The MLP is then refitted on all fit rows
and sigmoid-calibrated on the separate calibration role. Preprocessing and random oversampling
remain inside fit. Fixed controls isolate signed-log, robust scaling and oversampling effects.

The selected candidate is frozen before scoring retained validation. Additional partition seeds
and fit-size learning curves are descriptive checks with fixed parameters. Optuna reuses scoring
folds, so its best CV cost is a selection score with possible optimism, not an independent estimate.
At that optimization stage, the official test remained unused. See [the methodology](docs/optimization-methodology.md) and
[ADR 0003](docs/decisions/0003-bound-search-and-keep-mlp-stopping-inside-fit.md).

MLflow records a local SQLite parent run with trial/control children, parameters, metrics,
provenance and aggregate figures. Telemetry is disabled. After configuring the Windows terminal:

```powershell
uv run --no-sync mlflow server --backend-store-uri sqlite:///artifacts/tracking/mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000. Keep the server local. Data, tracking databases and trained models are
ignored by Git. On other platforms, set `MLFLOW_DISABLE_TELEMETRY=true` before starting the server.

The executed reference run selects tuned XGBoost: search CV cost 37,350 versus 38,740 for
the fixed XGBoost control. On retained validation, its frozen threshold produces 410 FP and
11 FN: cost 9,600, recall 94.5%, precision 31.55%. This lowers challenge cost while increasing
alerts; AP and Brier worsen versus the previous unweighted reference. The small-budget MLP
remains behind XGBoost on the three partition seeds. RobustScaler-only logistic fails convergence
under its common 100-iteration control budget and is explicitly excluded.
Measured results, tradeoffs and figures are recorded in [docs/results/optimization-03.md](docs/results/optimization-03.md).

## Checks and CI

```bash
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev python -m pytest
uv run --frozen --extra dev python -m fleetguard.smoke
uv run --frozen --extra dev python -m fleetguard.comparison_smoke
uv run --frozen --extra dev python -m build
```

For research verification, install `dev` and `research`, and run
`uv run --no-sync python -m fleetguard.optimization_smoke` too. Research tests skip when their
optional dependencies are absent; CI installs `dev`, `research` and `serve` for the complete suite.
Serving tests cover JSON contracts, startup integrity, body limits and prediction parity.
A separate container job builds the image and probes an explicit synthetic mounted bundle.

Linux CI runs on Python 3.11 and 3.12; a Windows Python 3.12 job runs the copy-mode
PowerShell checks. It tests CSV parsing, data integrity, grouped splits, cost
calculations, exact threshold search, preprocessing isolation, artifact reloads and batch inference.
It also tests separated roles, frozen calibration, champion selection independent of retained
validation labels, PyTorch epoch-role isolation, local tracking and XGBoost/MLP artifact reloads.
It builds a wheel and smoke-tests that wheel
outside the source directory. The synthetic smoke
fixture exercises the software offline and is never presented as a Scania benchmark.

The full UCI download and training run are deliberately excluded from pull-request CI. CI does
not require dataset hosting availability and does not spend compute on retraining for code-only
changes. Deployment automation will arrive with a deployable service.

## Repository layout

```text
configs/baseline.toml        Reproducible experiment settings
src/fleetguard/data.py       Official CSV parsing and data checks
src/fleetguard/download.py   UCI archive import and source integrity
src/fleetguard/splits.py     Duplicate-aware validation split
src/fleetguard/models.py     Reference model pipelines
src/fleetguard/metrics.py    Cost, metrics and exact threshold search
src/fleetguard/experiment.py Training and validation outputs
src/fleetguard/artifacts.py  Saved model contract and inference schema
src/fleetguard/evaluate.py   Frozen champion final-test evaluation
src/fleetguard/cli.py        User-facing commands
tests/                      Behavioral and integration tests
docs/                       Methodology, decisions and measured results
.github/workflows/ci.yml     Code and package verification
```

Raw data, fitted models, virtual environments and generated reports are ignored by Git.
Version the code, config, lockfile and small documented experiment summaries.

## Limits and next work

The original baseline selects thresholds and models on one development holdout. Version 0.2
adds train-only CV with separate calibration and threshold roles, but selecting among candidates
still induces optimism. The retained validation has already been inspected in the baseline.
Version 0.3 adds budgeted search, controlled ablations and an MLP. Search folds are reused for
selection, and the recorded stability concerns partition seeds with model seed fixed.
Version 0.4 completed the audits, CPU microbenchmark, freeze and one-shot final evaluation.
Version 0.5 serves that fixed decision and 0.6 supplies the local review interface.
Next: deployment and measured operations.

Duplicate grouping addresses only identical released feature rows. Anonymization prevents
physical explanations of individual sensors and does not allow us to rule out repeated trucks,
fleet overlap, selection bias or temporal leakage. Diagnostic performance on this historical
benchmark does not establish production reliability.

Only load trusted locally produced `joblib` artifacts. A checksum detects accidental changes;
it does not make a pickle safe to load from an unknown source.

The implementation plan is in [docs/roadmap.md](docs/roadmap.md), and the complete progress
checklist through portfolio publication is in [PROJECT_CHECKLIST.md](PROJECT_CHECKLIST.md).

## License and attribution

FleetGuard source code is released under the [MIT license](LICENSE). The dataset is not included
and retains its own terms. UCI currently displays CC BY 4.0; the source CSV preamble and description
include a GPL v3-or-later notice from Scania. Both are recorded in
[docs/dataset.md](docs/dataset.md); the code license does not replace either dataset notice.

## Frozen release (0.4)

Use a **complete official** update 03 run whose champion is uncalibrated `xgboost_tuned`:

```bash
uv run --no-sync python -m fleetguard audit --run artifacts/optimization/YOUR_COMPLETE_RUN
# Read the printed release model_card.md, decision.json and freeze.json.
uv run --no-sync python -m fleetguard evaluate-test --run artifacts/releases/YOUR_RELEASE --final
```

The audit reuses the base model and recorded roles. It compares raw/sigmoid via grouped calibration-role
OOF Brier, fits the winning map there, and tunes the challenge-cost threshold on the separate threshold
role. The pipeline/decision is frozen before retained-validation diagnostics and official-test scoring.
Inference verifies the freeze. No official-test threshold sweep or model adjustment is performed.
A receipt shared across runs in the same raw-data copy blocks accidental repeat evaluation attempts,
including interrupted ones. Do not delete receipts or `final_test` to retry a different model.
Legacy run evaluation remains supported; for the published release protocol, use the audited release.

Measured reference (official test, **16,000 rows / 375 APS positives**):

| FP | FN | Recall | Precision | Average precision | Brier | Challenge cost | Flagged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 475 | 16 | 95.73% | 43.05% | 0.88444 | 0.007538 | 12,750 | 5.21% |

The frozen threshold is **0.002490004285364349** on the sigmoid score scale. Source raw and release
make identical retained-validation decisions (410 FP / 11 FN), with improved validation Brier
0.013906 → 0.007209. Calibration does not establish reliable probabilities in another fleet.
The published official test has no exact feature-row overlap with official train. Challenge cost
units are not euros; dataset labels describe failure attribution, not healthy-versus-future-failing trucks.

Inspect [the complete model card](docs/results/release-04.md),
[aggregate evidence](docs/results/release-04.json), and [methodology](docs/release-methodology.md).
Local outputs include detailed FP/FN, missingness slices, per-column and grouped permutation audits,
inspection-budget scenarios, five figures and an isolated CPU microbenchmark. Only aggregates and
figures are versioned; source data, individual predictions and model weights remain local.

Run the offline contract smoke (core dependencies only):

```bash
uv run --no-sync python -m fleetguard.release_smoke
```

The API and subsequent UI/cloud increments serve this release. Since final outcomes are now published,
reusing the same test to improve a model cannot be presented as a fresh untouched evaluation.


## Local inference API (0.5)

Use your existing **complete frozen release**, without re-running optimization or official test.
On Windows, initialize every terminal with `. .\scripts\use-environment.ps1 -Research -Serve`.
On Linux/macOS:

```bash
uv sync --frozen --extra dev --extra serve
uv run --no-sync python -m fleetguard serve --release artifacts/releases/YOUR_RELEASE
```

Open <http://127.0.0.1:8000/docs>. The API exposes health/readiness, `/v1/model`,
`/v1/predict` and `/v1/predict-batch`. Every row needs all frozen sensor keys, each a finite
float32-compatible JSON number or explicit `null`. Default limits: 256 rows, 2 MiB body,
16 admitted connections/tasks and one inference worker slot. Model identity and threshold travel
with each prediction. Only a verified, warmed-up release becomes ready.

With the server running, in a second configured terminal:

```bash
uv run --no-sync python scripts/probe-api.py --release artifacts/releases/YOUR_RELEASE
uv run --no-sync python scripts/make-api-request.py --release artifacts/releases/YOUR_RELEASE --input sensors.csv --output reports/api-request.json
```

The probe compares HTTP scores against the local artifact. A sensor-only CSV can be converted to a
valid JSON batch without entering 170 values manually. Without `--input`, the helper writes one
explicit all-null contract probe; that is not a meaningful diagnosis. See [UPDATE_05.md](UPDATE_05.md)
for the PowerShell POST, bundle/Docker workflow and Git commands.

```bash
uv run --no-sync python -m fleetguard bundle --release artifacts/releases/YOUR_RELEASE --output artifacts/serving/YOUR_RELEASE
# Set FLEETGUARD_BUNDLE_DIR to that bundle's absolute path before running Compose.
docker compose up --build -d
# After verifying readiness, identity and parity:
docker compose down
```

The image runs as UID 10001 and installs locked core + serving dependencies without research
libraries. Compose mounts the bundle read-only; no models or raw data are baked into the image.
The HTTP reference check passed on four training rows from the frozen official release, with
unchanged hashes and decisions. It does not measure accuracy, production capacity or HTTP latency.
Docker execution and remote CI results remain to be confirmed locally/on GitHub.

Read [the complete v1 contract](docs/api-contract.md),
[ADR 0005](docs/decisions/0005-load-one-frozen-release-and-bound-http-inputs.md), and
[measured contract evidence](docs/results/api-05.json). This is a localhost service;
public hosting, authentication/CORS, model promotion and operations follow in later increments.


## Local web interface (0.6)

Use Node.js 24 LTS and the existing Python serving environment. Build from locked frontend dependencies:

```bash
cd frontend
npm ci
npm run check
npm test
npm run build
cd ..
uv run --no-sync python -m fleetguard serve --release artifacts/releases/YOUR_RELEASE --web-dir frontend/dist
```

Open <http://127.0.0.1:8000>. The three views cover CSV alert review, reference model diagnostics and
published CV experiments. Exact sensor schemas, finite float32 inputs and client caps are checked
before inference. Requests are split by server row/byte caps, with progress, cancellation and model
identity checks. Results are displayed only for a complete successful lot. Review annotations persist
across tabs in memory and can be deliberately exported with the model identity.

Uploaded data have no labels: the lot reports counts, alert fraction and missingness rather than
invented precision/recall/cost. Reference charts/tables clearly identify their published release,
independent of the model currently served. Examples are artificial null/zero contract probes. No
sensor records or models are shipped with the frontend.

`npm run dev` uses localhost:5173 and proxies the API routes to localhost:8000. The integrated built
UI uses the API origin directly. No public hosting or authentication/CORS policy is added yet.
The browser test uses an explicit synthetic HTTP fixture and the production build:

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

Windows setup, npm.cmd wrappers, execution-policy/session environment and the feature-branch workflow
are in [UPDATE_06.md](UPDATE_06.md). [Interface behavior and validation](docs/web-interface.md),
[ADR 0006](docs/decisions/0006-separate-unlabeled-review-from-reference-evidence.md), and the
[global checklist](PROJECT_CHECKLIST.md) describe the delivery and remaining cloud/portfolio stages.

Local validation is recorded in [web-06.json](docs/results/web-06.json), with
[desktop](docs/results/web-06-home.png) and [mobile](docs/results/web-06-mobile.png) captures.
These use the published reference release; the mobile lot contains only artificial null/zero inputs.
The interface displays the identity and threshold of the release actually loaded by your API.

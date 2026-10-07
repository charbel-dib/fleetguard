# FleetGuard

Cost-aware classification of Air Pressure System (APS) failures from Scania truck operational data.

An unnecessary inspection and a missed failure do not have the same cost. FleetGuard compares
simple reference classifiers with regularized logistic regression, then selects a decision
threshold using the challenge cost: **10 × false positives + 500 × false negatives**.

This repository starts with the data and evaluation foundation. It contains an installable Python
package, verified source snapshots, a reproducible holdout, saved inference pipelines and CI.
Tree ensembles, calibration, an API and a web application are subsequent increments.

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

These commands work in PowerShell too. The archive download is about 54 MiB. The complete training
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
prevent test reuse across different runs; preserving the test holdout remains an experimental rule.

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
results should not be interpreted as unbiased final-test estimates. The official test has not
been evaluated. Full metrics and source/environment provenance are in
[docs/results/baseline-validation.md](docs/results/baseline-validation.md).

## Checks and CI

```bash
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest
uv run --frozen --extra dev python -m fleetguard.smoke
uv run --frozen --extra dev python -m build
```

CI runs on Python 3.11 and 3.12. It tests CSV parsing, data integrity, grouped splits, cost
calculations, exact threshold search, preprocessing isolation, artifact reloads and batch inference.
It builds a wheel and smoke-tests that wheel outside the source directory. The synthetic smoke
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

The single validation fold is used both to choose thresholds and compare candidates, so its
selected results are optimistic development estimates. The next increment introduces
cross-validation within the training partition, tree models, controlled preprocessing ablations,
calibration and error analysis. The official test remains untouched during that work.

Duplicate grouping addresses only identical released feature rows. Anonymization prevents
physical explanations of individual sensors and does not allow us to rule out repeated trucks,
fleet overlap, selection bias or temporal leakage. Diagnostic performance on this historical
benchmark does not establish production reliability.

Only load trusted locally produced `joblib` artifacts. A checksum detects accidental changes;
it does not make a pickle safe to load from an unknown source.

The implementation plan is in [docs/roadmap.md](docs/roadmap.md).

## License and attribution

FleetGuard source code is released under the [MIT license](LICENSE). The dataset is not included
and retains its own terms. UCI currently displays CC BY 4.0; the source CSV preamble and description
include a GPL v3-or-later notice from Scania. Both are recorded in
[docs/dataset.md](docs/dataset.md); the code license does not replace either dataset notice.

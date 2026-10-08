# FleetGuard — frozen model card

Release: `20261007T210839Z-3b5571e8`. Frozen at `2026-10-07T21:08:40.510353+00:00`.
Base: `xgboost_tuned` from `20261007T184315Z-a4e5b402`; no base refit.
Calibration: **sigmoid**. Threshold: **0.00249000428536**.

## Task and intended use

Portfolio/research demonstration of cost-aware APS diagnostic classification on the public Scania snapshot.
pos means APS-associated failure; neg means another component failure, not a healthy truck.
The 170 measurements are anonymized; timestamps, vehicle identity and a forecasting horizon are absent.
This model is not validated for maintenance deployment, safety decisions or new fleet populations.

## Selection and separation

The base champion/parameters were selected by reused development scoring CV in update 03.
That search has selection optimism and is not nested CV. This release retains that base pipeline.
Raw versus sigmoid is selected by pooled Brier on grouped OOF predictions inside the calibration role.
The selected map is fit on that complete role. Both roles remain disjoint from the base fit and threshold role.
Calibration rows previously contributed to development hyperparameter search, so this is not independent model assessment.
A lower Brier combines reliability and discrimination; it does not prove universal probability calibration.
The threshold minimizes 10FP + 500FN on the threshold role; ties favor fewer FN, FP, then higher threshold.
The selected policy is challenge_cost. Validation diagnostics cannot change this decision.
Validation was observed in earlier increments; only the official test remains the final untouched benchmark.

| Calibration-role OOF method | Rows | Brier | Average precision |
|---|---:|---:|---:|
| raw | 9600 | 0.011309 | 0.861212 |
| sigmoid | 9600 | 0.005819 | 0.860233 |

## Measured outcomes

| Partition / pipeline | FP | FN | Recall | Precision | AP | Brier | Cost | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation / source raw | 410 | 11 | 94.50% | 31.55% | 0.813589 | 0.013906 | 9600 | 4.99% |
| Validation / release | 410 | 11 | 94.50% | 31.55% | 0.813589 | 0.007209 | 9600 | 4.99% |
| Official test / frozen release | 475 | 16 | 95.73% | 43.05% | 0.884442 | 0.007538 | 12750 | 5.21% |

Costs use challenge units, not euros. Validation and official test have different prevalence.
Official test rows: 16000; positives: 375.
Exact feature rows also found in official train: 0.
The official test was scored once after freeze; no model or threshold was revised from its outcomes.

95% conditional intervals (500 bootstrap resamples of complete duplicate-feature groups):
- cost_per_row: [0.594016, 1.056609]
- recall: [0.935778, 0.973894]
- precision: [0.394440, 0.461759]
- inspection_rate: [0.048813, 0.055569]
These intervals exclude model training/selection uncertainty and do not prove transport to other fleets.

## Inspection scenarios on retained validation

| Policy | Budget on threshold role | Actual validation flagged | Recall | Cost |
|---|---:|---:|---:|---:|
| challenge_cost | none | 4.99% | 94.50% | 9600 |
| budget_0.01 | 1% | 0.92% | 50.00% | 50100 |
| budget_0.03 | 3% | 2.92% | 87.00% | 14760 |
| budget_0.05 | 5% | 4.99% | 94.50% | 9600 |

These are frozen-threshold scenarios, not hard capacity guarantees on new batches.
The test uses only challenge_cost. The budget policies are not selected or compared using official-test labels.

## Error analysis and interpretation

Local validation_errors.csv records FP/FN row positions, scores, threshold margin and missingness.
error_sensor_summary.csv reports sensor medians/nonmissing counts by outcome, without inventing causal explanations.
Permutation importance uses every retained validation row and a fixed decision threshold; repeats measure shuffle variability.
It does not measure training uncertainty. Correlated sensors can mask or share importance; shuffling can break plausible combinations.
Multi-column prefixes are also shuffled jointly; these are naming groups, not identified physical subsystems.
No feature is removed, no model is refit and no threshold is selected from these diagnostics.

## CPU resource benchmark

Platform: Linux-6.18.44-x86_64-with-glibc2.39; Python 3.12.14; logical CPUs 9.
One thread; 20 measured repeats after one warm-up per batch.
Pipeline: 33,497 bytes; load: 0.1889 s.
Whole-process peak RSS: 193.81 MiB; traced Python peak: 11.16 MiB.
RSS includes interpreter, imports and native allocations; it is not the model's incremental memory.
Latency covers DataFrame-to-score only, excludes CSV/HTTP/network, and is specific to this environment.

| Batch | Median ms | p95 ms | Rows/s at median |
|---:|---:|---:|---:|
| 1 | 2.792 | 3.467 | 358 |
| 128 | 3.028 | 3.412 | 42266 |
| 1024 | 4.760 | 4.943 | 215138 |

## Artifact and limitations

The persisted pipeline includes preprocessing, optional calibration, ordered schema and runtime versions.
freeze.json hashes the pipeline, metadata, decision, policies, configuration, source and role manifests.
Runtime verifies that freeze before inference/final evaluation. These hashes detect changes; they are not signatures.
Only load trusted locally created joblib artifacts. Data/model files remain local and are not committed.
The final-evaluation receipt is shared across runs using the same raw-data directory and prevents accidental repeat scoring.
It is a local guard, not an access-control system; deleting receipts or copying raw data can bypass it.
Release monitoring, drift thresholds, production SLAs and external validation remain future work.

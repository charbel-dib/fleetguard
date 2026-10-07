# First complete validation benchmark

Executed: 2026-10-07. Run ID: `20261007T122016Z-1ecc15cc`.

This run uses the complete official Scania training file and the committed default configuration.
It was executed with the packages selected by `uv.lock` for Python 3.12. The original archive
was downloaded from UCI and imported through `fleetguard download --archive`.

## Partitions and source checks

- Official train: 60,000 observations, 170 features, 1,000 positives.
- Fitting partition: 48,000 observations, 800 positives.
- Validation: 12,000 observations, 200 positives.
- Seed: 42; five stratified group folds; validation fold: 0.
- Exact duplicate feature rows in the official train: 0.
- Missing sensor cells: 8.3335% of the training table.
- Constant observed feature: `cd_000`; no entirely missing feature.
- Official test: structurally validated at import, not evaluated.

The archive SHA-256 is
`5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89`.
All source member hashes, the split hash, environment versions and full metrics are recorded in
[baseline-validation.json](baseline-validation.json). The numerical table is also available as
[baseline-validation.csv](baseline-validation.csv).

## Results

| Model | Rule | Threshold | Average precision | Recall | Precision | FP | FN | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Always negative | Fixed | 0.5 | 0.0167 | 0.0000 | 0.0000 | 0 | 200 | 100,000 |
| Class prior | Selected | Above constant score | 0.0167 | 0.0000 | 0.0000 | 0 | 200 | 100,000 |
| Logistic | 0.5 | 0.5 | 0.8008 | 0.6450 | 0.8600 | 21 | 71 | 35,710 |
| Logistic | Selected | 0.0408842 | 0.8008 | 0.9050 | 0.4458 | 225 | 19 | 11,750 |
| Logistic balanced | 0.5 | 0.5 | 0.7658 | 0.9000 | 0.3956 | 275 | 20 | 12,750 |
| Logistic balanced | Selected | 0.328777 | 0.7658 | 0.9250 | 0.3033 | 425 | 15 | 11,750 |

At its selected threshold, unweighted logistic regression misses 19 failures and triggers 225
unnecessary inspections. Its cost is `19 × 500 + 225 × 10 = 11,750`, an 88.25% reduction from
the always-negative reference on this validation set. It flags 406 observations: 181 true
positives and 225 false positives, or 3.3833% of the holdout.

The balanced model misses four fewer failures but adds 200 false positives. These changes exactly
offset at the specified costs. The unweighted model wins the declared tie-breaker on average
precision. A different cost assumption could prefer a different threshold or candidate.

## Interpretation and limits

Changing the unweighted model's decision threshold produces a larger cost improvement here than
its threshold-0.5 result suggests. The additional recall reduces missed-failure cost, while the
precision falls because more observations are flagged. These are measured consequences of a
decision rule, not a general claim that low thresholds or unweighted models are always better.

Both thresholds and candidate selection use this same holdout. Their selected metrics are
optimistic development estimates. No confidence intervals, cross-validated model selection,
probability calibration, temporal validation or physical sensor explanations are claimed.
The next increment should test whether the differences survive more rigorous comparisons.

The serialized artifacts and per-observation predictions are generated locally under `artifacts/`
and intentionally excluded from Git. Rerunning `fleetguard train` recreates them with a new run ID.
Fit timing is recorded for inspection, not as a hardware-normalized speed benchmark.

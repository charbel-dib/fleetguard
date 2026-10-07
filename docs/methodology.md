# Evaluation protocol — increment 1

## Target and unit of observation

One observation is one row of released operational measurements. The positive class identifies
a specific APS component failure; the negative class identifies failures elsewhere. The target
is encoded as `pos → 1`, `neg → 0`. The released data do not support a forecast horizon or
vehicle-level train/test grouping.

## Source validation

Import the complete official UCI archive. Preserve all three source files unchanged and record
the archive and member SHA-256 hashes, URL, DOI and retrieval timestamp. Validate train/test
column agreement, the number of rows, the 170 sensor columns, accepted labels, numeric values
and absence of infinities. The train must contain exactly 1,000 positives.

The source preamble is located by finding the `class,...` header; no fixed `skiprows=20` is assumed.
Duplicate column names are checked before pandas can rename them. Feature names are not inferred
to have physical meanings, and the nonstandard names `am_0` and `ec_00` are retained.

Download/import parses the test file to validate its structural contract. Training and profiling
commands do not compute test distributions or metrics, nor fit anything on test observations.

## Holdout construction

Hash the complete numerical feature row, excluding the target and pandas index, using
`pandas.util.hash_pandas_object`. Duplicate rows, including duplicates with conflicting labels,
receive the same 64-bit group. Hash collisions are theoretically possible; this grouping is a
practical exact-row check rather than a physical vehicle identity.

Use five-fold `StratifiedGroupKFold` with shuffle enabled and seed 42. Hold out fold 0. Persist
every positional row ID, group hash and assignment. Check that partitions cover all rows, contain
both classes and have no group overlap. Class ratio and fold size are approximately preserved.
The environment snapshot and lockfile matter: library changes can alter split implementations.

No holdout observation contributes to preprocessing statistics. The official test file remains
separate throughout model and threshold development.

## Reference models

| Model | Fitted state | Decision rule |
|---|---|---|
| Always negative | Class labels only | Always `neg` via threshold 0.5 on zero score |
| Class prior | Training positive prevalence | Threshold 0.5 and validation-selected rule |
| Logistic | Medians, missing-indicator schema, scaling and coefficients | Both rules |
| Logistic balanced | Same components with class weights computed on train | Both rules |

For the logistic pipelines, `SimpleImputer(strategy='median', add_indicator=True,
keep_empty_features=True)` imputes each feature. An entirely missing training column is retained
with value zero. Indicators are created only for features missing at fit time; a newly missing
feature at inference is imputed but does not gain a new indicator.

`StandardScaler` centers and scales the imputed features and indicators. `LogisticRegression`
uses binary `liblinear`, default L2 regularization, `C=1`, `tol=1e-4`, `max_iter=2000`.
The balanced variant applies inverse-frequency training weights. No clipping, resampling,
feature selection, hyperparameter search or calibration is performed in this increment.
Convergence warnings fail the experiment and leave a failed run marker.

## Threshold and candidate selection

Prediction is `pos` when `score >= threshold`. Sort validation scores in descending order,
accumulate true and false positives at the end of each tied-score group, and compute all distinct
decision rules. Add an all-negative rule using `nextafter(max_score, +∞)`; if the maximum is one,
the sentinel threshold is slightly above one. Include the lowest observed score for all-positive.

Choose the rule minimizing `10 × FP + 500 × FN`. Break cost ties by fewer false negatives,
then fewer false positives, then the highest threshold. Persist the entire threshold curve.
The always-negative baseline keeps its fixed rule; other models use their selected threshold.

Select the candidate with the lowest validation cost, then highest average precision, then
alphabetical model name. These tie rules are explicit for deterministic comparison, not additional
hyperparameters. Save both the selected and threshold-0.5 metrics for every model.

Thresholds and candidate choice share a validation partition. Their selected metrics are optimistic
development estimates. Cross-validation of model choices belongs in the next increment; neither
the current validation nor the smoke fixture should be described as final-test performance.

## Metrics

- **Primary:** total challenge cost, accompanied by its FP and FN counts.
- **Ranking:** average precision and ROC-AUC. Average precision is not trapezoidal PR-AUC.
- **Decisions:** recall, precision, F1, accuracy, balanced accuracy and inspection rate.
- **Scores:** Brier score as an initial diagnostic, not evidence of calibrated probabilities.

Always show prevalence and sample counts. Undefined precision/recall is recorded as zero when
there are no predicted/actual positives; the reusable metric function returns null ranking metrics
for single-class input. Threshold tuning requires both classes.

## Frozen final evaluation

`evaluate-test --final` verifies the source snapshot, loads the saved validation-selected champion,
restores the feature order and applies its frozen threshold and costs. It does not refit the model,
evaluate competing candidates on the test or select a new threshold. It records official test
feature-row overlap with the official training file without removing those observations.

The command records one final evaluation per run and refuses to overwrite it. This filesystem
check is not a guarantee against experimental misuse: generating another run and looking at test
results repeatedly still contaminates the holdout.

## Reproducibility boundary

Each run records configuration, source hashes, split assignments and hash, environment versions,
model metadata, pipeline hash and validation predictions. Reproducibility means fixed data,
environment and decisions. Fit timing and generated run IDs vary. Numerical results may vary
slightly with CPU libraries or operating systems; bit-for-bit cross-platform equality is not claimed.

Source and methodology references:

- [UCI APS dataset](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
- [scikit-learn: common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html)
- [StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [SimpleImputer](https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html)
- [LogisticRegression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)

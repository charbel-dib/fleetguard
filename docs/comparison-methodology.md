# Train-only comparison protocol — increment 2

This protocol compares ten fixed-budget candidates. It introduces more reliable threshold
evaluation, two preprocessing ablations, tree ensembles, a calibration experiment and scored
error analysis. Hyperparameter optimization and neural tabular models are subsequent work.

## Partitions

The initial split retains the increment-1 configuration: seed 42, five stratified group folds,
fold 0 held out. The original 60,000-row training file yields 48,000 development observations
and 12,000 retained validation observations. Identical numerical feature rows share a group.

The retained validation was already inspected in increment 1. It is therefore a development
check, not an untouched blind test. The official UCI test is not evaluated by `compare`.

Within development, three stratified group CV folds use seed 43. Each fold reserves one third
for scoring. Its remaining observations are partitioned again into five group folds:

| Role | Default fraction of that fold's pool | Use |
|---|---:|---|
| Fitting | Approximately 60% | Fit imputers, scalers and the candidate estimator |
| Calibration | Approximately 20% | Fit the sigmoid calibrator for the HGB calibration candidate |
| Threshold | Approximately 20% | Select the minimum-cost threshold |
| Scoring | Held out from the pool | Evaluate the frozen candidate and threshold |

In the default Scania run, a CV fitting role contains approximately 19,200 observations.
No observation in a scoring role contributes to that fold's preprocessing, estimator,
calibrator or threshold. Every development observation is scored once per candidate across
the outer CV folds. All candidates use the same roles in a given fold.

The calibration role is reserved for every candidate to equalize fitting sample sizes. It is
unused by uncalibrated candidates. This is a sample-efficient starting compromise, not the
only valid calibration design. It deliberately sacrifices fitting data to keep the roles clear.

`cv_roles.csv` stores the global source row ID, feature group, CV fold and role. `final_roles.csv`
stores the roles for the deployment candidate. Their hashes join the original split hash in
`run.json`; inference and final-test evaluation reject modified role manifests.

## Candidates

| Candidate | Input handling | Training choice |
|---|---|---|
| `always_negative` | No transform | Fixed negative decision rule |
| `class_prior` | No transform | Constant training prevalence score |
| `logistic` | Median, missing indicators, standard scaling | Unweighted L2 logistic |
| `logistic_balanced` | Same | Inverse-frequency training weights |
| `logistic_no_indicator` | Median and standard scaling | Missing-indicator ablation |
| `logistic_log_robust` | Signed log, median, indicators, robust scaling | Heavy-tail preprocessing variant |
| `random_forest` | Median and indicators; no scaling | Balanced subsample weights, bounded depth |
| `hist_gradient_boosting` | Native numerical missing values | Fixed histogram-boosting budget |
| `hgb_sigmoid` | The same already-fitted HGB | Sigmoid calibration on the separate role |
| `xgboost` | Native numerical missing values | CPU histogram boosting, fixed budget |

Signed log is `sign(x) * log1p(abs(x))`, defined for negative and positive observations and
preserving missing values. It is applied before median imputation and has no learned statistics.
The log/robust candidate changes two preprocessing choices together; it does not isolate the
effect of robust scaling alone. This combined variant is named explicitly to avoid that claim.

HGB's internal random early-stopping holdout is disabled. Training lengths are configured
before the comparison. No early stopping or estimator tuning reads a scoring fold. The HGB
calibration variant reuses the fitted HGB through scikit-learn `FrozenEstimator`, so calibration
does not retrain its trees. Calibration quality is assessed on scored observations using Brier
score and reliability bins; it is not assumed to improve merely because calibration was fitted.

The CPU-only XGBoost distribution is used on Windows/Linux. This run does not require CUDA.
Threads are limited to the configured budget, including OpenMP/BLAS calls through threadpoolctl.

## Thresholds, metrics and selection

For each candidate/fold, the threshold role selects the rule minimizing `10 FP + 500 FN` using
the exact tied-score-aware search introduced in increment 1. The always-negative baseline keeps
threshold 0.5. Scoring labels never enter this selection. The report records metrics for both
the frozen threshold and 0.5 on the scoring role.

Primary candidate selection uses total scoring cost divided by total scored observations.
Cost ties use mean fold average precision, then model name. Mean fold ranking metrics are
reported rather than treating uncalibrated scores from different fitted models as one common
probability scale. Pooled out-of-fold predictions are retained for inspection.

Standard deviations describe variation across the three correlated folds. They are not confidence
intervals or proof of a statistically significant difference. Model selection still compares
multiple candidates using the same CV results; the selected CV performance can be optimistic.

## Final development candidate

Choose the champion from CV **before** predicting the retained 12,000 validation observations.
Partition the 48,000 development observations into fitting/calibration/threshold roles again
using seed 43. In the default run, these contain 28,800/9,600/9,600 observations respectively.
Fit only the champion and any base estimator required for calibration. Preserve that exact
model, calibrator and threshold in the saved artifact.

Score only that champion on retained validation. Do not choose another candidate or retune its
threshold based on those metrics. Saving the champion and its model/input contract keeps the
existing `predict` and `evaluate-test --final` commands usable. Official test evaluation remains
a separate final action after experimental choices are fixed.

Increment-1 models fit on 48,000 observations and select thresholds on the retained validation.
Increment-2 models use smaller fitting roles and separate threshold tuning. Their numbers should
not be placed in a before/after table as if only the estimator changed.

## Error and reliability outputs

- `cv_fold_metrics.csv`: scored fold metrics for both decision rules.
- `cv_summary.csv`: candidate selection table, fold means and standard deviations.
- `oof_predictions.csv`: every scored development row, candidate, fold and frozen threshold.
- `validation_metrics.json`: the single frozen champion's retained validation metrics.
- `reliability.csv`: occupied equal-width bins with observation counts and positive fractions.
- `errors_by_missingness.csv`: scored errors for 0–5%, 5–20%, 20–50% and 50–100% missing sensors.
- `error_cases.csv`: retained-validation false positives and false negatives, with source row IDs.
- `figures/`: CV cost, precision/recall, confusion counts and reliability plots.

Missingness groups are diagnostics; their differences are not causal explanations. Single-class
segments have null ranking metrics. Reliability plots need their bin counts for interpretation;
sparse bins can be unstable, particularly with this class imbalance.

## References

- [StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [CalibratedClassifierCV and FrozenEstimator](https://scikit-learn.org/1.8/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost CPU installation](https://xgboost.readthedocs.io/en/stable/install.html)

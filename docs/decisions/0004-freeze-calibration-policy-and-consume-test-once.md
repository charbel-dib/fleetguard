# ADR 0004 — calibrate the fixed champion and freeze before final evaluation

Status: accepted before official-test scoring (update 04).

## Context

The update 03 XGBoost champion minimizes the bounded development search cost, but its balanced
training weights yield poorly scaled positive scores. The retained validation has already been
observed. We need a portable inference artifact and honest final evidence without adapting to test.

## Decision

1. Reuse the actual uncalibrated `xgboost_tuned` champion, including its fit preprocessing,
   parameters and training rows. Do not refit or select a different base model in this increment.
2. Read its recorded split/final-role manifests; verify row coverage and exact feature groups.
3. Within its separate calibration role, compare raw scores with sigmoid calibration using three
   grouped calibration folds, seed 46. Choose pooled held-out Brier, raw winning ties.
   Then fit the chosen map on the complete calibration role. Reject reversed sigmoid rankings.
4. Select the decision threshold on the separate threshold role only, minimizing 10FP + 500FN,
   with the established tie-break. The selected policy is the unconstrained challenge cost.
   Threshold-role inspection budgets of 1%, 3%, 5% are descriptive alternative scenarios.
5. Save the complete inference pipeline and a timestamped decision, then hash source, roles,
   settings, policies, model and metadata. Perform validation diagnostics only after this freeze.
6. Evaluate the official test once using the frozen champion/policy. No test policy sweep,
   candidate comparison, threshold adjustment, feature removal or model refit follows scoring.
7. Claim a receipt shared by runs under the same raw-data parent before reading test labels.
   An interrupted attempt remains consumed; its receipt must not be removed to obtain a new score.

## Consequences

Calibration CV selects only a mapping; the base hyperparameter search had already used all
48,000 development rows, including the final calibration role, in CV. It is not independent
assessment of model selection. Brier combines reliability, discrimination and prevalence.
Different calibrators per OOF fold can change pooled ranks; final sigmoid is monotone.

A fixed threshold constrained on the threshold role does not enforce capacity on new batches.
Permutation diagnostics use observed validation and never feed feature/model selection.
Exact duplicates are grouped, but unknown truck/time dependence cannot be excluded.

The final receipt is an accidental-repeat guard local to this snapshot directory; copying data
or deleting receipts can bypass it. It is not a security boundary or proof of unknown external use.
Legacy train/compare/optimize evaluation remains compatible, but now also consumes this shared receipt.
For the published project evaluation, use the released artifact and preserve the receipt.

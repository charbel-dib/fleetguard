# 0002 — Separate calibration, threshold tuning and scoring

Status: accepted for increment 2.

## Problem

Increment 1 selected thresholds and candidates on the same validation observations. That was a
useful baseline, but the reported selected costs were optimistic. Adding more models without
changing the protocol would increase the opportunity to overfit that development holdout.

## Decision

Retain the same outer validation assignment. Compare fixed candidates using three group CV folds
inside the development partition. Divide each fold's pool into fitting, calibration and threshold
roles. Fit transforms and estimators on fitting only, fit a frozen HGB sigmoid calibrator on
calibration only, tune thresholds on the threshold role, and score on the outer CV fold.

Choose the champion by scoring-fold cost before evaluating retained validation. Train its final
artifact with the same role design. Preserve complete manifests and metrics, plus diagnostics
of reliability and errors by missingness. Keep baseline commands and artifact schema compatible.

## Consequences

Costs can now be evaluated on observations that did not select the corresponding threshold.
The calibration comparison uses a frozen base estimator and separate labels. Exact duplicate
groups do not cross roles. The protocol remains inspectable and runnable on CPU.

Reserving calibration and threshold observations reduces fitting sample sizes. This is a deliberate
cost of a simple, explicit design. Increment-1 metrics are retained as historical results and are
not treated as a comparable same-data model benchmark.

Three fold standard deviations describe variation, not statistical significance. Choosing among
multiple candidates still induces selection optimism. Further tuning must remain inside development
and the official test must stay reserved for the final frozen experimental decision.

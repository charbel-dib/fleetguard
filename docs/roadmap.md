# FleetGuard — next code increments

This is an implementation sequence, not a time estimate. Each increment should leave the
repository runnable and add evidence for the decisions it introduces.

## 1. Data and baseline foundation — this commit

Implemented: source import, validation, grouped holdout, two reference classifiers, two logistic
variants, exact cost-aware threshold selection, run outputs, pipeline persistence, batch inference,
tests and package CI. The final-test command exists but is not part of development runs.

## 2. Model comparison and error analysis

- Add `training/cross_validation.py`: train-only stratified group folds, identical folds for all
  candidates, out-of-fold predictions, fold metrics and uncertainty across folds.
- Extend `models.py` with Random Forest, HistGradientBoosting and XGBoost. Give each candidate
  its own preprocessing contract; trees do not need standardized inputs.
- Add `features/ablations.py`: missing indicators on/off, robust scaling, optional train-only
  clipping and transformation of suitable nonnegative counters.
- Add `evaluation/calibration.py`: out-of-fold/split-separated calibration and reliability plots.
  Avoid using the same labels to fit calibration and report its quality.
- Add `evaluation/errors.py`: FP/FN cases, errors by missingness and stability across seeds.
- Add Optuna and MLflow only with explicit search spaces, compute budgets and logged split hashes.
- Document selected choices in `docs/experiments.md`, with failed candidates and real results.

Acceptance: compare every candidate with the same data protocol and report whether gains persist
across folds. Preserve the current validation and official test roles.

## 3. Inference service

- Add `serving/predictor.py`: typed feature schema, loaded artifact and score/decision contract.
- Add `api/main.py`, `/health`, `/ready`, `/predict`, `/predict-batch`; use FastAPI/Pydantic.
- Set request limits, reject unknown feature fields, expose model version and log structured errors.
- Add API contract tests and a container image; compare API scores with the saved local pipeline.
- Benchmark CPU inference and batch throughput with a fixed request dataset.

Acceptance: a versioned service starts from a trusted artifact, predicts reproducibly, and fails
clearly on bad input or unavailable models.

## 4. Web application

- Build a React/TypeScript frontend around batch upload and inspection review.
- Add a threshold control with counts and cost computed from validation observations, clearly
  distinguished from unlabeled production inputs.
- Provide a model comparison view, missingness diagnostics and an FP/FN explorer.
- Add loading/error states, accessibility checks, API integration tests and a production build.

Acceptance: a user can run an example, inspect an alert, and understand the measured tradeoffs.
Do not invent sensor meanings or interpret uncalibrated scores as probabilities.

## 5. Deployment and portfolio integration

- Add Docker builds and registry publication, staging verification and deployment workflows.
- Declare hosting and artifact storage configuration, model compatibility and rollback procedure.
- Add service metrics and a fixed load benchmark; choose alert thresholds from measured behavior.
- Deploy the frontend and inference service, then integrate the demo and experiment evidence
  into the Vercel portfolio.
- Perform the frozen official-test evaluation after the experimental choices are finalized.

Acceptance: the published demo identifies its model, dataset, runtime limits, measured quality
and hosting conditions. CI/CD should exercise a real service and deployment path.

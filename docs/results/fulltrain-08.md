# FleetGuard release 08 — full-train protocol (`release-08`)

Model and threshold were frozen from the training file only (2026-10-31T18:09:19.275785+00:00); the official test was scored once afterwards.

## Official test (16,000 rows, 375 APS failures)

| Model | Cost (10·FP + 500·FN) | False alarms | Missed failures |
|---|---:|---:|---:|
| **FleetGuard release 08 (this run)** | **8,530** | 553 | 6 |
| IDA 2016 winner (Costa & Nascimento) | 9,920 | 542 | 9 |
| IDA 2016 second place (Gondek, Hafner & Sampson) | 10,900 | 490 | 12 |

- Recall **98.40%** (95% bootstrap: 97.06%–99.49%)
- Precision 40.02% (95% bootstrap: 36.77%–43.30%)
- Cost 95% bootstrap interval: 6,410–11,160
- Flagged trucks: 5.76%

A missed failure costs as much as 50 false alarms, so the cost-optimal policy
flags widely: low precision is the intended trade-off, not a defect.

## How the threshold was chosen

Grouped 5-fold cross-validation on all 60,000 training rows (identical sensor rows stay in one fold). The threshold `0.00341056` minimizes the challenge cost on out-of-fold scores: cost 36,860, 37 missed failures and 1836 false alarms on training.

## Model

Average of three histogram gradient boosting classifiers (scikit-learn), seeds [0, 1, 2], positive class weight 20, 400 iterations, learning rate 0.05. Hyperparameters were fixed a priori, not tuned on the official test.

## Limits

- One public, anonymized snapshot: results need validation before use on another fleet.
- Scores are ranking scores, not calibrated probabilities.
- Reference costs come from the published challenge results; protocols may differ.

# ADR 0008: Full-train protocol for release 08

Status: accepted.

Releases 03–04 split the 60,000 official training rows into fit, calibration and threshold roles.
The frozen model therefore learned from only part of the 1,000 positives, and the official-test cost
(12,750) stayed above the published IDA 2016 results (9,920 and 10,900).

Release 08 keeps a single boundary instead. Grouped 5-fold cross-validation on the full training
file produces out-of-fold scores; identical sensor rows stay in one fold, as in earlier splits.
The decision threshold minimizes 10·FP + 500·FN on those scores. The final model, an average of three
class-weighted histogram gradient boosting classifiers, is then fit on all training rows and frozen
with its threshold and checksums before the official test is read. `fulltrain-test --final` refuses
to score a release twice.

Hyperparameters were fixed a priori (400 iterations, learning rate 0.05, 31 leaves, L2 1.0, positive
weight 20), not searched. Calibration is not applied: the cost decision needs a ranking and a
threshold, not probabilities.

Disclosure: before this protocol was written, one exploratory run with the same model family but
ungrouped folds and a coarser threshold grid was evaluated once on the official test (cost 9,490).
Grouped folds were adopted because they are the repository's standard against duplicate leakage,
not because of that score. No setting was changed after the release 08 test evaluation.

The official test has now been used by releases 04 and 08. Further model work must be judged with
cross-validation on the training file; a new official-test number would no longer be untouched.

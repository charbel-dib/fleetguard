# Release audit and final evaluation (update 04)

This stage uses only core dependencies. It accepts the uncalibrated `xgboost_tuned` champion
of a **complete** update 03 optimization run. If another family wins your run, the command fails
explicitly; do not modify champion.json to bypass that restriction. Retain that result and review
its protocol before extending this audit to an already calibrated model.

## Roles and selection

The original split.csv and final_roles.csv are reused verbatim. The audit validates row coverage,
feature hashes and duplicate-group separation, rather than reconstructing roles from new labels.
For the measured Scania source: base fit 28,800; calibration 9,600; threshold 9,600; retained
validation 12,000. No base preprocessing or classifier is refit in this increment.

Within calibration, three StratifiedGroupKFold partitions (seed 46) compare raw predictions to
sigmoid maps trained on the other calibration folds using FrozenEstimator. The criterion is
pooled held-out Brier; raw wins ties. The winning map is fit on all calibration rows. Sigmoid
must preserve ranking (a strictly decreasing sklearn sigmoid coefficient); reversed maps fail.
The base model predictions are checked to remain identical after calibration fitting.

The base hyperparameter search in update 03 already used development CV including these rows.
Thus calibration fit is disjoint from base fit, but these data are not independent of all prior
model selection. Pooled calibration OOF AP can differ from raw AP because folds fit distinct maps.
A single final monotone map preserves ranks unless numerical saturation creates ties.

The selected operating policy is fixed as unconstrained challenge cost: 10FP + 500FN.
Exact score ties stay together; equal cost favors fewer FN, FP, then the higher threshold.
Budgets 1%, 3%, 5% constrain flagged fraction **on the threshold role only**. They are evaluated
on retained validation as descriptive alternatives, and are never swept on the official test.
They do not enforce a hard capacity limit on new batches.

A calibrated score does not imply 0.5 is an appropriate decision threshold. Even the theoretical
Bayes threshold 10/(10+500) assumes exact population probabilities and stable costs/distribution.
The empirical threshold here is fitted only on its assigned role; no population guarantee is claimed.

## Freeze and descriptive diagnostics

Before reading validation labels for this audit, decision.json records the source artifact hash,
parameters, calibration method, policy, threshold and UTC time. freeze.json hashes ten files:
source, config, split/final/calibration-role manifests, champion, decision, policies and model/metadata.
run.json pins the freeze hash when the complete audit succeeds. Inference/evaluation verifies it.
Changing metadata, roles or the pipeline makes the release fail integrity checks.
Hashes identify content; they are not signatures or protection against a trusted user rewriting files.

Validation was observed in previous development increments. The following diagnostics cannot
select features, hyperparameters, calibration or threshold:

- Full-row permutation audit, 170 columns × three shuffles, seed 46; fixed-threshold cost per row
  and average precision are reported together. Three shuffles give limited precision.
- Joint permutation of the seven multi-column naming prefixes, preserving relationships inside
  each shuffled block. Prefixes are not known physical systems.
- Local FP/FN records with score, margin and missing fraction; outcome sensor medians and counts.
- Error slices at missingness [0,5%), [5,20%), [20,50%), [50,100%]. Interpret positive counts:
  a slice with zero APS labels cannot establish recall.
- Reliability bins and frozen policy tradeoffs. Lower Brier combines discrimination/reliability
  and prevalence; it alone does not prove calibration in another population.

Permutation can create implausible sensor combinations; correlation can hide or share importance.
Its standard deviation measures shuffle variability, not model training uncertainty or causality.
Do not turn these diagnostics into maintenance explanations for individual trucks.

## CPU benchmark

A dedicated process loads the complete release and receives at most 1,024 fit-role sensor rows.
It measures load time, then one warm-up and 20 timed repeats for batches 1/128/1024 with thread
pools limited to one. Timed repeats run without Python allocation tracing. Latency covers an
already parsed DataFrame to scores, excluding CSV parsing, HTTP and network. It is not a concurrency/load test.
It is a local microbenchmark, not a service SLA or a measurement on the user's GPU/PC.

The native-memory measure is whole-process peak RSS: Linux /proc/self/status VmHWM, Windows
GetProcessMemoryInfo PeakWorkingSetSize, macOS getrusage (bytes). The Linux method avoids including
a larger parent before exec. Python tracemalloc separately records the maximum observed peak
of model loading or a warm-up. RSS includes imports and native buffers; the 33-KB pipeline file
is not evidence that the inference process needs only 33 KB of RAM.

## Official final test

The existing explicit `evaluate-test --final` command uses the release's frozen model and threshold,
without comparing alternative policies. It verifies the official snapshot and release integrity
before claiming a receipt in data/final_evaluations/<snapshot-key>/, shared across run directories.
The claim is an exclusive directory creation **before test label loading**. A completed or failed
attempt consumes the receipt. An interruption does not grant permission to try another artifact.
The receipt stores source test hash, model hash, threshold, freeze hash and status.

Legacy run evaluation remains supported with the same shared guard. Per-run final_test output is
also immutable. This local mechanism cannot detect runs in other copies of the data directory or
previous evaluations outside the project; preserve its state and report previous use honestly.

The final output includes metrics, local predictions, source overlap and 95% percentile intervals
from 500 bootstrap samples (seed 47) of entire identical-feature groups. These are conditional on
the fixed model/snapshot, exclude training/selection uncertainty, and do not model unknown truck/time
clustering. The observed official test contains 16,000 distinct feature groups and no exact overlap
with official train. No data are removed from the challenge test because of overlap checking.

Published results and the complete model card are in [results/release-04.md](results/release-04.md).
They are reproduction evidence. Once these results are public, reusing this test to iterate on
models cannot be described as a fresh untouched evaluation. Subsequent increments serve and deploy
this release; a future ML revision requires an additional credible external evaluation protocol.

References: [scikit-learn 1.8 calibration](https://scikit-learn.org/1.8/modules/calibration.html),
[permutation importance](https://scikit-learn.org/1.8/modules/permutation_importance.html),
[ADR 0004](decisions/0004-freeze-calibration-policy-and-consume-test-once.md).

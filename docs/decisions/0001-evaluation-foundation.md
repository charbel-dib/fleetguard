# 0001 — Establish a cost-aware evaluation foundation

Status: accepted for increment 1.

## Problem

Only a small fraction of official training rows are positive. Optimizing or displaying accuracy
alone can hide a classifier that misses every APS failure. Preprocessing and threshold selection
can also contaminate a holdout if their learned state uses validation or test observations.

## Decision

Start with two trivial references and unweighted/weighted logistic regression. Hold out one
stratified group fold from the official training file, grouping duplicate feature rows. Fit
preprocessing inside scikit-learn pipelines. Select thresholds and candidates by validation cost.
Save complete pipelines, input schema, threshold, source provenance and validation predictions.

Use TOML and a small CLI rather than a workflow framework at this stage. Save each run in a
distinct local directory. Dependency resolution is locked with uv. CI uses synthetic fixtures
and verifies the packaged wheel independently of the editable installation.

Keep final-test evaluation in a separate explicit command. Do not add API or frontend shells
before an inference contract and baseline exist.

## Consequences

The first comparison is inexpensive and inspectable. Every learned transform and threshold has
a clear source partition. Exact score ties and error costs can be tested exhaustively.

A single development holdout is a starting point, not a robust final model-selection protocol.
Class weighting does not solve calibration. Local JSON/CSV run tracking lacks a shared experiment
registry; MLflow can be introduced when the next increment expands the comparison.

Duplicate grouping is an available-data safeguard, not evidence of vehicle-level independence.
The final-test command discourages accidental repeat evaluation but cannot enforce methodological
discipline across newly created runs.

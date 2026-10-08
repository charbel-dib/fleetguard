"""Conditional percentile intervals, resampling complete duplicate-feature groups."""

import numpy as np

from fleetguard.metrics import validate_scores


def grouped_intervals(target, scores, groups, *, threshold, costs, seed=47, repeats=500):
    y, p = validate_scores(target, scores)
    groups = np.asarray(groups)
    if groups.shape != y.shape or repeats < 2:
        raise ValueError("Bootstrap groups must match rows and repeats must be at least two.")
    unique, inverse = np.unique(groups, return_inverse=True)
    positive = p >= threshold
    counts = np.array(
        [
            np.bincount(inverse, weights=mask, minlength=len(unique))
            for mask in (
                (positive & (y == 1)),
                (positive & (y == 0)),
                (~positive & (y == 1)),
                np.ones(len(y)),
            )
        ]
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        tp, fp, fn, rows = counts @ weights
        values.append(
            [
                (fp * costs["false_positive_cost"] + fn * costs["false_negative_cost"]) / rows,
                tp / (tp + fn) if tp + fn else np.nan,
                tp / (tp + fp) if tp + fp else np.nan,
                (tp + fp) / rows,
            ]
        )
    matrix = np.asarray(values)
    return {
        "method": "percentile bootstrap of complete identical-feature groups",
        "coverage": 0.95,
        "seed": seed,
        "repeats": repeats,
        "groups": len(unique),
        "scope": "conditional on this fixed model and snapshot; excludes training/selection "
        "uncertainty",
        "intervals": {
            name: {
                "lower": float(np.nanquantile(matrix[:, column], 0.025)),
                "upper": float(np.nanquantile(matrix[:, column], 0.975)),
                "valid_replicates": int(np.isfinite(matrix[:, column]).sum()),
            }
            for column, name in enumerate(
                ("cost_per_row", "recall", "precision", "inspection_rate")
            )
            if np.isfinite(matrix[:, column]).any()
        },
    }

"""Reliability and error summaries computed on scored, never tuning, observations."""

import numpy as np
import pandas as pd

from fleetguard.metrics import classification_metrics, validate_scores


def reliability_table(target, scores, *, bins: int = 10) -> pd.DataFrame:
    y, p = validate_scores(target, scores)
    if bins < 1:
        raise ValueError("bins must be positive.")
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for number in range(bins):
        mask = bucket == number
        if mask.any():
            rows.append(
                {
                    "bin": number,
                    "lower": number / bins,
                    "upper": (number + 1) / bins,
                    "count": int(mask.sum()),
                    "mean_score": float(p[mask].mean()),
                    "positive_rate": float(y[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def error_by_missingness(
    target, scores, missing_fraction, *, threshold: float, costs: dict
) -> pd.DataFrame:
    y, p = validate_scores(target, scores)
    missing = np.asarray(missing_fraction, dtype=float)
    if (
        missing.shape != p.shape
        or not np.isfinite(missing).all()
        or ((missing < 0) | (missing > 1)).any()
    ):
        raise ValueError("Missing fractions must be finite fractions matching the scores.")
    edges = [0.0, 0.05, 0.20, 0.50, np.nextafter(1.0, np.inf)]
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (missing >= lower) & (missing < upper)
        if mask.any():
            metrics = classification_metrics(y[mask], p[mask], threshold=threshold, **costs)
            rows.append(
                {
                    "missing_lower": lower,
                    "missing_upper": min(upper, 1.0),
                    "positive": int(y[mask].sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)

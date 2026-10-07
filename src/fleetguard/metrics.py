"""Cost-aware metrics and exact threshold search, with score ties kept together."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_scores(target, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Targets and scores must be nonempty, equal-length vectors.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary (0/1).")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Scores must be finite values between 0 and 1.")
    return y.astype(int), p


def classification_metrics(
    target,
    scores,
    *,
    threshold: float,
    false_positive_cost: int = 10,
    false_negative_cost: int = 500,
) -> dict:
    y, p = validate_scores(target, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, np.inf):
        raise ValueError("Threshold must be in [0, nextafter(1, +inf)].")
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("Error costs must be positive.")
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    cost = fp * false_positive_cost + fn * false_negative_cost
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction))
        if both_classes
        else None,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "brier_score": float(brier_score_loss(y, p)),
        "cost": int(cost),
        "cost_per_row": float(cost / len(y)),
        "false_positive_cost": int(false_positive_cost),
        "false_negative_cost": int(false_negative_cost),
        "inspection_rate": float(prediction.mean()),
    }


def threshold_curve(
    target, scores, *, false_positive_cost: int = 10, false_negative_cost: int = 500
) -> pd.DataFrame:
    y, p = validate_scores(target, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold tuning requires both classes.")
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("Error costs must be positive.")
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    # Each endpoint includes every observation having the same score.
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(p) - 1]
    cumulative_tp = np.cumsum(sorted_y)[ends]
    cumulative_fp = ends + 1 - cumulative_tp
    thresholds = np.r_[np.nextafter(sorted_p[0], np.inf), sorted_p[ends]]
    tp = np.r_[0, cumulative_tp]
    fp = np.r_[0, cumulative_fp]
    fn = int(y.sum()) - tp
    tn = int((y == 0).sum()) - fp
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "cost": fp * false_positive_cost + fn * false_negative_cost,
        }
    )


def select_threshold(curve: pd.DataFrame) -> float:
    # Equal cost: fewer missed failures, then fewer inspections, then highest threshold.
    best = curve.sort_values(
        ["cost", "fn", "fp", "threshold"], ascending=[True, True, True, False]
    ).iloc[0]
    return float(best["threshold"])

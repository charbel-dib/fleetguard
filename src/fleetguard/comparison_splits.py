"""Disjoint fitting, calibration, threshold tuning and scoring roles."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(features, target, *, n_splits: int, seed: int):
    if not features.index.equals(target.index):
        raise ValueError("Feature and target indices differ.")
    if set(target.unique()) != {0, 1} or target.value_counts().min() < n_splits:
        raise ValueError("Each binary class must have at least n_splits observations.")
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(features, target, groups))
    for train, heldout in folds:
        if set(groups[train]) & set(groups[heldout]):
            raise ValueError("A feature group crossed a fold boundary.")
        if len(set(target.iloc[train])) != 2 or len(set(target.iloc[heldout])) != 2:
            raise ValueError("Every fold partition must contain both classes.")
    return folds


def role_split(
    features, target, *, n_splits: int, seed: int, calibration_fold: int, threshold_fold: int
) -> tuple[dict, pd.DataFrame]:
    if n_splits < 3 or calibration_fold == threshold_fold:
        raise ValueError("Fitting, calibration and threshold tuning need distinct roles.")
    if not (0 <= calibration_fold < n_splits and 0 <= threshold_fold < n_splits):
        raise ValueError("Role fold index out of range.")
    folds = grouped_folds(features, target, n_splits=n_splits, seed=seed)
    calibration = folds[calibration_fold][1]
    tuning = folds[threshold_fold][1]
    fitting = np.setdiff1d(np.arange(len(features)), np.r_[calibration, tuning])
    roles = {"fit": fitting, "calibration": calibration, "threshold": tuning}
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    labels = np.full(len(features), "fit", dtype=object)
    for name, indices in roles.items():
        if len(set(target.iloc[indices])) != 2:
            raise ValueError(f"The {name} partition must contain both classes.")
        labels[indices] = name
    for left, right in (("fit", "calibration"), ("fit", "threshold"), ("calibration", "threshold")):
        if set(groups[roles[left]]) & set(groups[roles[right]]):
            raise ValueError("A duplicate feature group crossed roles.")
    assignments = pd.DataFrame(
        {"local_row": np.arange(len(features)), "feature_group": groups, "role": labels}
    )
    return roles, assignments

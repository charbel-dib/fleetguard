"""Hold out one stratified group fold from the official training file."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int,
    n_splits: int,
    validation_fold: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(features) != len(target) or not features.index.equals(target.index):
        raise ValueError("Features and target must have matching rows and indices.")
    if set(target.unique()) != {0, 1} or target.value_counts().min() < n_splits:
        raise ValueError("Each binary class must have at least n_splits examples.")
    if not 0 <= validation_fold < n_splits:
        raise ValueError("validation_fold is outside n_splits.")
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    training, validation = list(splitter.split(features, target, groups))[validation_fold]
    for indices in (training, validation):
        if len(set(target.iloc[indices])) != 2:
            raise ValueError("Split has a single-class partition; change the split configuration.")
    if set(groups[training]) & set(groups[validation]):
        raise ValueError("Duplicate feature groups crossed the split boundary.")
    assignments = pd.DataFrame(
        {"row_id": np.arange(len(features)), "feature_group": groups, "partition": "train"}
    )
    assignments.loc[validation, "partition"] = "validation"
    return training, validation, assignments

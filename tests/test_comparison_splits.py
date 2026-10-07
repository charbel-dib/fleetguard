import numpy as np
import pandas as pd
import pytest

from fleetguard.comparison_splits import grouped_folds, role_split


def test_role_partitions_are_complete_and_groups_are_disjoint(classification_frame):
    features, target = classification_frame
    features = pd.concat([features, features.iloc[:10]], ignore_index=True)
    target = pd.concat([target, target.iloc[:10]], ignore_index=True)
    kwargs = {"n_splits": 5, "seed": 43, "calibration_fold": 0, "threshold_fold": 1}
    roles, assignments = role_split(features, target, **kwargs)
    roles_again, _ = role_split(features, target, **kwargs)
    assert sorted(np.concatenate(list(roles.values()))) == list(range(len(features)))
    for left, right in (("fit", "calibration"), ("fit", "threshold"), ("calibration", "threshold")):
        assert set(roles[left]).isdisjoint(roles[right])
        assert set(assignments.iloc[roles[left]]["feature_group"]).isdisjoint(
            assignments.iloc[roles[right]]["feature_group"]
        )
    for name in roles:
        np.testing.assert_array_equal(roles[name], roles_again[name])


def test_cv_scores_every_row_once(classification_frame):
    features, target = classification_frame
    folds = grouped_folds(features, target, n_splits=3, seed=43)
    scored = np.concatenate([heldout for _, heldout in folds])
    assert sorted(scored) == list(range(len(features)))
    assert len(set(scored)) == len(features)


def test_role_split_rejects_shared_or_invalid_roles(classification_frame):
    features, target = classification_frame
    with pytest.raises(ValueError, match="distinct roles"):
        role_split(features, target, n_splits=5, seed=43, calibration_fold=0, threshold_fold=0)
    with pytest.raises(ValueError, match="out of range"):
        role_split(features, target, n_splits=5, seed=43, calibration_fold=0, threshold_fold=5)

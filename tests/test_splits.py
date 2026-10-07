import numpy as np
import pandas as pd

from fleetguard.splits import make_split


def test_split_is_reproducible_complete_and_disjoint(classification_frame):
    frame, target = classification_frame
    kwargs = {"seed": 42, "n_splits": 5, "validation_fold": 0}
    train, validation, assignments = make_split(frame, target, **kwargs)
    train_again, validation_again, _ = make_split(frame, target, **kwargs)
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(validation, validation_again)
    assert set(train).isdisjoint(validation)
    assert sorted(np.r_[train, validation]) == list(range(len(frame)))
    assert set(assignments["partition"]) == {"train", "validation"}


def test_identical_feature_rows_do_not_cross_partitions(classification_frame):
    frame, target = classification_frame
    # Deliberately include duplicate measurements with a conflicting label.
    duplicates = frame.iloc[:20].copy()
    combined = pd.concat([frame, duplicates], ignore_index=True)
    labels = pd.concat([target, 1 - target.iloc[:20]], ignore_index=True)
    train, validation, assignments = make_split(
        combined, labels, seed=42, n_splits=5, validation_fold=0
    )
    assert set(assignments.iloc[train]["feature_group"]).isdisjoint(
        assignments.iloc[validation]["feature_group"]
    )
    for i in range(20):
        assert assignments.iloc[i]["partition"] == assignments.iloc[len(frame) + i]["partition"]

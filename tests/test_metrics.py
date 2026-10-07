import numpy as np
import pytest

from fleetguard.metrics import classification_metrics, select_threshold, threshold_curve


def test_error_cost_is_assigned_to_the_right_confusion_entries():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], threshold=0.5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["cost"] == 510


def test_high_accuracy_can_hide_missed_failures():
    result = classification_metrics([0] * 99 + [1], [0.0] * 100, threshold=0.5)
    assert result["accuracy"] == 0.99
    assert result["recall"] == 0.0
    assert result["cost"] == 500


@pytest.mark.parametrize("seed", range(10))
def test_exact_threshold_search_matches_exhaustive_predictions(seed):
    rng = np.random.default_rng(seed)
    y = np.r_[0, 1, rng.integers(0, 2, 38)]
    scores = rng.choice([0.0, 0.1, 0.2, 0.5, 0.9, 1.0], len(y))
    curve = threshold_curve(y, scores)
    for row in curve.itertuples(index=False):
        prediction = scores >= row.threshold
        fp = int(((y == 0) & prediction).sum())
        fn = int(((y == 1) & ~prediction).sum())
        assert row.fp == fp
        assert row.fn == fn
        assert row.cost == 10 * fp + 500 * fn
    chosen = select_threshold(curve)
    assert classification_metrics(y, scores, threshold=chosen)["cost"] == int(curve["cost"].min())


def test_all_negative_and_all_positive_rules_are_candidates():
    curve = threshold_curve([0, 1], [1.0, 0.0])
    assert (curve.iloc[0][["fp", "tp"]] == 0).all()
    assert (curve.iloc[-1][["fn", "tn"]] == 0).all()


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [np.inf, 0.5], [-0.1, 0.5], [0.0, 1.1]])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        threshold_curve([0, 1], scores)


def test_one_class_metrics_are_serializable_but_threshold_tuning_is_rejected():
    result = classification_metrics([0, 0], [0.1, 0.3], threshold=0.5)
    assert result["roc_auc"] is None
    assert result["average_precision"] is None
    with pytest.raises(ValueError, match="both classes"):
        threshold_curve([0, 0], [0.1, 0.3])

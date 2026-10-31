import numpy as np
import pandas as pd
import pytest

from fleetguard.fulltrain import (
    COSTS,
    FullTrainConfig,
    SeedEnsemble,
    cost_threshold,
    feature_groups,
    outcome,
)


def brute_force(target, scores):
    best = None
    for t in [*np.unique(scores), np.inf]:
        result = outcome(target, scores >= t)
        key = (result["cost"], result["fn"], result["fp"], -t)
        if best is None or key < best[0]:
            best = (key, t, result["cost"])
    return best[1], best[2]


@pytest.mark.parametrize("seed", range(5))
def test_cost_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    target = rng.random(300) < 0.08
    scores = np.round(rng.random(300) * 0.5 + target * rng.random(300) * 0.5, 2)
    threshold, cost = cost_threshold(target, scores)
    expected_threshold, expected_cost = brute_force(target, scores)
    assert cost == expected_cost == outcome(target, scores >= threshold)["cost"]
    assert (
        outcome(target, scores >= threshold)["fn"]
        == outcome(target, scores >= expected_threshold)["fn"]
    )


def test_flag_nothing_when_no_positives():
    threshold, cost = cost_threshold(np.zeros(10, bool), np.linspace(0, 1, 10))
    assert cost == 0 and threshold > 1


def test_outcome_uses_challenge_costs():
    result = outcome([1, 1, 0, 0], [1, 0, 1, 0])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["cost"] == COSTS["false_positive"] + COSTS["false_negative"]


def test_duplicate_rows_share_a_group():
    frame = pd.DataFrame({"a": [1.0, 1.0, np.nan, np.nan], "b": [2.0, 2.0, 3.0, 4.0]})
    groups = feature_groups(frame)
    assert groups[0] == groups[1] and groups[2] != groups[3]


def test_seed_ensemble_averages_members(classification_frame):
    features, target = classification_frame
    config = FullTrainConfig(seeds=(0, 1), max_iter=20)
    model = SeedEnsemble(seeds=config.seeds, max_iter=config.max_iter).fit(features, target)
    used = features[model.used_columns_]
    expected = np.mean([m.predict_proba(used) for m in model.models_], axis=0)
    assert "empty" not in model.used_columns_
    np.testing.assert_allclose(model.predict_proba(features), expected)
    assert list(model.feature_names_in_) == list(features.columns)

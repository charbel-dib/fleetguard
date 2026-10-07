from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import fleetguard.comparison as comparison
from fleetguard.artifacts import load_model, positive_scores
from fleetguard.comparison_config import load_comparison_config
from fleetguard.comparison_smoke import smoke_config
from fleetguard.experiment import require_complete
from fleetguard.io import read_json
from fleetguard.splits import make_split


def test_comparison_is_complete_and_validation_labels_do_not_select_the_model(
    tmp_path, monkeypatch
):
    x, y = make_classification(
        n_samples=600, n_features=10, n_informative=6, weights=[0.85, 0.15], random_state=42
    )
    features, target = pd.DataFrame(x), pd.Series(y)
    features.columns = [f"sensor_{i}" for i in features.columns]
    config = replace(
        smoke_config(tmp_path), models=("always_negative", "logistic", "hgb_sigmoid", "xgboost")
    )
    development, validation, assignments = make_split(
        features, target, seed=42, n_splits=5, validation_fold=0
    )
    # Freeze the stratified split first. Perturb only the scoring labels afterwards.
    monkeypatch.setattr(
        comparison, "make_split", lambda *args, **kwargs: (development, validation, assignments)
    )
    run = comparison.run_comparison(
        features, target, config, source={"dataset": "fixture"}, kind="synthetic_smoke"
    )
    changed = target.copy()
    changed.iloc[validation] = 1 - changed.iloc[validation]
    altered_run = comparison.run_comparison(
        features, changed, config, source={"dataset": "fixture"}, kind="synthetic_smoke"
    )
    assert require_complete(run)["status"] == "complete"
    assert not (run / "final_test").exists()
    champion = read_json(run / "champion.json")
    assert champion == read_json(altered_run / "champion.json")
    assert champion["selection_partition"] == "development_cross_validation"
    name = champion["model"]
    original_metadata = read_json(run / name / "metadata.json")
    changed_metadata = read_json(altered_run / name / "metadata.json")
    assert original_metadata["threshold"] == changed_metadata["threshold"]
    original_model, _ = load_model(run / name)
    changed_model, _ = load_model(altered_run / name)
    np.testing.assert_allclose(
        positive_scores(original_model, features.iloc[validation]),
        positive_scores(changed_model, features.iloc[validation]),
    )
    oof = pd.read_csv(run / "oof_predictions.csv")
    assert len(oof) == len(development) * len(config.models)
    assert not oof[["model", "row_id"]].duplicated().any()
    assert set(oof["row_id"]).isdisjoint(validation)
    role_assignments = pd.read_csv(run / "cv_roles.csv")
    assert set(role_assignments["row_id"]).isdisjoint(validation)
    for _, fold in role_assignments.groupby("cv_fold"):
        assert len(fold) == len(development)
        assert fold.groupby("feature_group")["role"].nunique().max() == 1
    assert (run / "figures/reliability.png").is_file()
    assert (run / "error_cases.csv").is_file()
    scored = pd.read_csv(run / "cv_summary.csv")
    assert scored.iloc[0]["model"] == name
    with (run / "final_roles.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="role manifest checksum mismatch"):
        require_complete(run)


def test_comparison_config_preserves_original_outer_split():
    from pathlib import Path

    from fleetguard.config import load_config

    original = load_config(Path("configs/baseline.toml"))
    newer = load_comparison_config(Path("configs/comparison.toml"))
    assert (original.seed, original.n_splits, original.validation_fold) == (
        newer.base.seed,
        newer.base.n_splits,
        newer.base.validation_fold,
    )

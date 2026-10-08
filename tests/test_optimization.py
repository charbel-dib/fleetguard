import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fleetguard.artifacts import load_model, positive_scores
from fleetguard.experiment import require_complete
from fleetguard.io import read_json
from fleetguard.splits import make_split


@pytest.fixture
def research():
    for name in ("torch", "optuna", "imblearn", "mlflow"):
        # Disable before the optional MLflow import as well as inside LocalTracker.
        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
        pytest.importorskip(name)


def test_search_ignores_retained_validation_and_tracks_local_runs(tmp_path, monkeypatch, research):
    from mlflow.telemetry import get_telemetry_client

    import fleetguard.optimization as optimization
    from fleetguard.optimization_smoke import research_smoke_config
    from fleetguard.tracking import LocalTracker

    assert get_telemetry_client() is None
    x, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, weights=[0.85, 0.15], random_state=42
    )
    features, target = pd.DataFrame(x), pd.Series(y)
    features.columns = [f"sensor_{i}" for i in features.columns]
    config = replace(
        research_smoke_config(tmp_path), stability_seeds=(43, 44), learning_fractions=(0.5, 1.0)
    )
    development, validation, assignments = make_split(
        features, target, seed=42, n_splits=5, validation_fold=0
    )
    monkeypatch.setattr(
        optimization, "make_split", lambda *args, **kwargs: (development, validation, assignments)
    )
    run = optimization.run_optimization(
        features, target, config, source={"dataset": "fixture"}, kind="synthetic_smoke"
    )
    changed = target.copy()
    changed.iloc[validation] = 1 - changed.iloc[validation]
    other_config = replace(
        config,
        comparison=replace(
            config.comparison, base=replace(config.comparison.base, runs_dir=tmp_path / "other")
        ),
    )
    altered = optimization.run_optimization(
        features, changed, other_config, source={"dataset": "fixture"}, kind="synthetic_smoke"
    )
    assert require_complete(run)["status"] == "complete"
    assert read_json(run / "champion.json") == read_json(altered / "champion.json")
    assert read_json(run / "selected_parameters.json") == read_json(
        altered / "selected_parameters.json"
    )
    champion = read_json(run / "champion.json")["model"]
    original, meta = load_model(run / champion)
    repeated, meta2 = load_model(altered / champion)
    assert meta["threshold"] == meta2["threshold"]
    np.testing.assert_allclose(
        positive_scores(original, features.iloc[validation]),
        positive_scores(repeated, features.iloc[validation]),
        rtol=0,
        atol=0,
    )
    assert not (run / "final_test").exists()
    roles = pd.read_csv(run / "cv_roles.csv")
    assert set(roles["row_id"]).isdisjoint(validation)
    for _, fold in roles.groupby("cv_fold"):
        assert len(fold) == len(development)
        assert fold.groupby("feature_group")["role"].nunique().max() == 1
    epoch = pd.read_csv(run / "candidates/mlp_tuned/epoch_roles.csv")
    for fold, rows in epoch.groupby("cv_fold"):
        fit = roles[(roles["cv_fold"] == fold) & (roles["role"] == "fit")]
        assert set(rows["row_id"]) <= set(fit["row_id"])
        assert rows.groupby("feature_group")["role"].nunique().max() == 1
    trials = pd.read_csv(run / "trials.csv")
    assert trials.groupby("family").size().to_dict() == {"mlp": 1, "xgboost": 1}
    tracker = LocalTracker(config.tracking_dir)
    for row in trials.itertuples(index=False):
        tracked = tracker.client.get_run(row.mlflow_run_id)
        assert tracked.info.status == "FINISHED"
        assert tracked.data.metrics["cost_per_row"] == row.cost_per_row
    for path in (run / "candidates").glob("*/oof_predictions.csv"):
        oof = pd.read_csv(path)
        assert len(oof) == len(development)
        assert not oof["row_id"].duplicated().any()
    diagnostics = pd.read_csv(run / "diagnostic_roles.csv")
    assert set(diagnostics["row_id"]).isdisjoint(validation)
    for _, rows in diagnostics.groupby(["model", "diagnostic", "seed", "fraction", "cv_fold"]):
        assert rows.groupby("feature_group")["role"].nunique().max() == 1
    with (run / "final_roles.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="role manifest checksum mismatch"):
        require_complete(run)


def test_unconverged_control_is_recorded_and_cannot_be_published(tmp_path, monkeypatch, research):
    from sklearn.exceptions import ConvergenceWarning

    import fleetguard.optimization as optimization
    from fleetguard.optimization_smoke import research_smoke_config

    x, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, weights=[0.85, 0.15], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    original = optimization.evaluate_cv

    def fail_control(*args, **kwargs):
        if args[3] == "logistic_robust":
            raise ConvergenceWarning("fixture: iteration budget exhausted")
        return original(*args, **kwargs)

    monkeypatch.setattr(optimization, "evaluate_cv", fail_control)
    run = optimization.run_optimization(
        features,
        pd.Series(y),
        research_smoke_config(tmp_path),
        source={"dataset": "fixture"},
        kind="synthetic_smoke",
    )
    assert require_complete(run)["status"] == "complete"
    status = pd.read_csv(run / "control_status.csv").set_index("model")
    assert status.loc["logistic_robust", "status"] == "failed_convergence"
    assert "logistic_robust" not in pd.read_csv(run / "cv_summary.csv")["model"].tolist()
    assert (run / "candidates/logistic_robust/failure.json").exists()
    assert not (run / "logistic_robust/pipeline.joblib").exists()


def test_interrupted_search_marks_run_and_tracking_failed(tmp_path, monkeypatch, research):
    import fleetguard.optimization as optimization
    from fleetguard.optimization_smoke import research_smoke_config
    from fleetguard.tracking import LocalTracker

    x, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, weights=[0.85, 0.15], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    config = research_smoke_config(tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("fixture")

    monkeypatch.setattr(optimization, "make_candidate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        optimization.run_optimization(
            features, pd.Series(y), config, source={"dataset": "fixture"}, kind="synthetic_smoke"
        )
    run = next(config.comparison.base.runs_dir.iterdir())
    assert read_json(run / "run.json")["status"] == "failed"
    with pytest.raises(ValueError, match="complete training run"):
        require_complete(run)
    tracker = LocalTracker(config.tracking_dir)
    records = tracker.client.search_runs([tracker.experiment_id])
    assert len(records) == 2
    assert all(record.info.status == "FAILED" for record in records)

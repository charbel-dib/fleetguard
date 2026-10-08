from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fleetguard.artifacts import load_model, positive_scores, save_model
from fleetguard.comparison_splits import role_split
from fleetguard.experiment import require_complete
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.metrics import select_threshold, threshold_curve
from fleetguard.release import choose_calibration, policy_thresholds, run_audit, validated_roles
from fleetguard.release_config import ReleaseConfig, load_release_config
from fleetguard.splits import make_split
from fleetguard.uncertainty import grouped_intervals


@pytest.fixture
def source_release(tmp_path):
    x, y = make_classification(
        n_samples=800, n_features=8, n_informative=5, weights=[0.8, 0.2], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    # Exact duplicates must stay together in every audit boundary.
    features.iloc[-2:] = features.iloc[:2].to_numpy()
    target = pd.Series(y)
    development, _, split = make_split(features, target, seed=42, n_splits=5, validation_fold=0)
    x_dev, y_dev = features.iloc[development], target.iloc[development]
    roles, assignments = role_split(
        x_dev, y_dev, n_splits=5, seed=43, calibration_fold=0, threshold_fold=1
    )
    assignments["row_id"] = development
    assignments = assignments.drop(columns="local_row")
    parent = tmp_path / "source"
    parent.mkdir()
    split.to_csv(parent / "split.csv", index=False)
    assignments.to_csv(parent / "final_roles.csv", index=False)
    # Satisfy the previous protocol's saved role/checksum contract.
    assignments.to_csv(parent / "cv_roles.csv", index=False)
    model = Pipeline(
        [
            (
                "classifier",
                XGBClassifier(
                    n_estimators=20,
                    max_depth=2,
                    learning_rate=0.2,
                    n_jobs=1,
                    random_state=42,
                ),
            )
        ]
    )
    model.fit(x_dev.iloc[roles["fit"]], y_dev.iloc[roles["fit"]])
    scores = positive_scores(model, x_dev.iloc[roles["threshold"]])
    threshold = select_threshold(threshold_curve(y_dev.iloc[roles["threshold"]], scores))
    import xgboost

    save_model(
        parent / "xgboost_tuned",
        model,
        metadata={
            "model": "xgboost_tuned",
            "calibration": None,
            "threshold": threshold,
            "feature_names": features.columns.tolist(),
            "training_rows": len(roles["fit"]),
            "costs": {"false_positive_cost": 10, "false_negative_cost": 500},
            "parameters": {"n_estimators": 20, "max_depth": 2},
            "xgboost_version": xgboost.__version__,
        },
    )
    write_json(parent / "source.json", {"dataset": "synthetic fixture"})
    write_json(parent / "champion.json", {"model": "xgboost_tuned"})
    write_json(
        parent / "run.json",
        {
            "status": "complete",
            "kind": "synthetic_smoke",
            "protocol": "optimization_v1",
            "split_sha256": sha256_file(parent / "split.csv"),
            "cv_roles_sha256": sha256_file(parent / "cv_roles.csv"),
            "final_roles_sha256": sha256_file(parent / "final_roles.csv"),
        },
    )
    config = ReleaseConfig(
        raw_dir=tmp_path / "raw",
        releases_dir=tmp_path / "releases",
        calibration_folds=2,
        permutation_repeats=1,
        benchmark_repeats=2,
        batch_sizes=(1, 16),
    )
    return features, target, parent, config


def test_release_decision_ignores_validation_labels_and_preserves_base(source_release):
    x, y, parent, config = source_release
    model_hash = sha256_file(parent / "xgboost_tuned/pipeline.joblib")
    run = run_audit(x, y, config, parent)
    roles = validated_roles(x, parent)
    changed = y.copy()
    changed.iloc[roles["validation"]] = 1 - changed.iloc[roles["validation"]]
    altered = run_audit(
        x, changed, replace(config, releases_dir=config.releases_dir / "altered"), parent
    )
    one, two = read_json(run / "decision.json"), read_json(altered / "decision.json")
    for value in (one, two):
        value.pop("frozen_at_utc")
    assert one == two
    assert one["official_test_used_for_selection"] is False
    assert one["base_model_refitted"] is False
    assert require_complete(run)["protocol"] == "release_v1"
    assert model_hash == sha256_file(parent / "xgboost_tuned/pipeline.joblib")
    a, meta = load_model(run / "xgboost_release")
    b, _ = load_model(altered / "xgboost_release")
    np.testing.assert_array_equal(positive_scores(a, x), positive_scores(b, x))
    assert meta["threshold_source"] == "development_threshold_role"
    assert read_json(run / "validation_metrics.json") != read_json(
        altered / "validation_metrics.json"
    )
    cv = pd.read_csv(run / "calibration_roles.csv")
    assert set(cv.row_id) == set(roles["calibration"])
    for _, fold in cv.groupby("cv_fold"):
        merged = fold.merge(pd.read_csv(parent / "final_roles.csv"), on="row_id")
        assert merged.groupby("feature_group").role_x.nunique().max() == 1
    assert not (run / "final_test").exists()
    assert not (run / "benchmark_input.csv").exists()
    bench = read_json(run / "benchmark.json")
    assert bench["process_peak_rss_bytes"] > 0
    assert all(row["median_ms"] > 0 for row in bench["batches"])
    # Frozen metadata edits are rejected before inference/evaluation.
    path = run / "xgboost_release/metadata.json"
    metadata = read_json(path)
    metadata["threshold"] = 0.5
    write_json(path, metadata)
    with pytest.raises(ValueError, match="freeze checksum"):
        require_complete(run)


def test_calibrator_uses_only_recorded_calibration_rows(source_release):
    x, y, parent, config = source_release
    roles = validated_roles(x, parent)
    model, _ = load_model(parent / "xgboost_tuned")
    cal = roles["calibration"]
    result = choose_calibration(model, x.iloc[cal], y.iloc[cal], cal, config)
    changed = y.copy()
    changed.iloc[roles["threshold"]] = 1 - changed.iloc[roles["threshold"]]
    repeated = choose_calibration(model, x.iloc[cal], changed.iloc[cal], cal, config)
    assert result[1] == repeated[1]
    pd.testing.assert_frame_equal(result[2], repeated[2])
    np.testing.assert_array_equal(positive_scores(result[0], x), positive_scores(repeated[0], x))


def test_budget_thresholds_keep_ties_together():
    y = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.9, 0.6, 0.4, 0.4, 0.1])
    policies, _ = policy_thresholds(
        y, scores, {"false_positive_cost": 10, "false_negative_cost": 500}, (0.1, 0.5)
    )
    limited = policies[policies.policy != "challenge_cost"]
    assert (limited.inspection_rate <= limited.budget_on_threshold_role).all()
    assert limited.iloc[0].inspection_rate == 0
    for row in policies.itertuples():
        assert row.inspection_rate == np.mean(scores >= row.threshold)


def test_bootstrap_resamples_complete_groups_and_is_reproducible():
    kwargs = dict(
        threshold=0.5, costs={"false_positive_cost": 10, "false_negative_cost": 500}, repeats=20
    )
    result = grouped_intervals([0, 1, 1], [0.6, 0.6, 0.2], [7, 7, 7], **kwargs)
    assert result == grouped_intervals([0, 1, 1], [0.6, 0.6, 0.2], [7, 7, 7], **kwargs)
    assert result["groups"] == 1
    assert result["intervals"]["cost_per_row"]["lower"] == 170
    assert result["intervals"]["cost_per_row"]["upper"] == 170


def test_release_config_and_roles_reject_invalid_inputs(source_release):
    x, _, parent, config = source_release
    assert load_release_config(__import__("pathlib").Path("configs/release.toml")).seed == 46
    with pytest.raises(ValueError, match="calibration folds"):
        replace(config, calibration_folds=1)
    with pytest.raises(ValueError, match="budgets"):
        replace(config, inspection_budgets=(float("nan"),))
    changed = x.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="feature groups"):
        validated_roles(changed, parent)


def test_final_release_scores_only_frozen_policy(source_release, monkeypatch):
    import fleetguard.evaluate as evaluation
    from fleetguard.config import Config
    from fleetguard.data import TEST_NAME

    x, y, parent, config = source_release
    manifest = {"files": {TEST_NAME: "fixture-release-test"}}
    write_json(parent / "source.json", manifest)
    state = read_json(parent / "run.json")
    state["kind"] = "official_aps"
    write_json(parent / "run.json", state)
    run = run_audit(x, y, config, parent)
    frozen = sha256_file(run / "freeze.json")
    decision = read_json(run / "decision.json")
    monkeypatch.setattr(evaluation, "verify_raw", lambda _: manifest)
    # Synthetic loader boundary only: this is not an official Scania measurement.
    monkeypatch.setattr(evaluation, "load_aps", lambda _: (x.iloc[:50], y.iloc[:50]))

    def forbid_fit(*args, **kwargs):
        raise AssertionError("No model or calibrator may fit on final evaluation.")

    monkeypatch.setattr(XGBClassifier, "fit", forbid_fit)
    from sklearn.calibration import CalibratedClassifierCV

    monkeypatch.setattr(CalibratedClassifierCV, "fit", forbid_fit)
    output = evaluation.evaluate_test(
        Config(raw_dir=config.raw_dir, runs_dir=config.releases_dir), run
    )
    result = read_json(output / "metrics.json")
    assert result["threshold_source"] == "development_threshold_role"
    assert result["metrics"]["threshold"] == decision["threshold"]
    assert result["freeze_sha256"] == frozen
    assert sha256_file(run / "freeze.json") == frozen
    assert read_json(run / "decision.json") == decision
    assert read_json(output / "status.json")["status"] == "complete"
    assert "Official test / frozen release" in (run / "model_card.md").read_text()

import numpy as np
import pandas as pd
import pytest

import fleetguard.evaluate as evaluation
from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.config import Config
from fleetguard.data import TEST_NAME
from fleetguard.experiment import run_experiment
from fleetguard.io import read_json
from fleetguard.metrics import classification_metrics


def test_final_evaluation_uses_frozen_artifact_and_costs(
    tmp_path, classification_frame, monkeypatch
):
    features, target = classification_frame
    manifest = {"files": {TEST_NAME: "fixture-hash"}}
    original = Config(raw_dir=tmp_path / "raw", runs_dir=tmp_path / "runs")
    run = run_experiment(features, target, original, source=manifest)
    changed = Config(
        raw_dir=original.raw_dir,
        runs_dir=original.runs_dir,
        false_positive_cost=1,
        false_negative_cost=1,
    )
    monkeypatch.setattr(evaluation, "verify_raw", lambda _: manifest)
    # This fixture simulates the official loader boundary; it is not a real APS test score.
    monkeypatch.setattr(evaluation, "load_aps", lambda _: (features.iloc[:30], target.iloc[:30]))
    name = read_json(run / "champion.json")["model"]
    pipeline, metadata = load_model(run / name)
    expected = classification_metrics(
        target.iloc[:30],
        positive_scores(pipeline, prepare_input(features.iloc[:30], metadata["feature_names"])),
        threshold=metadata["threshold"],
        **metadata["costs"],
    )
    output = evaluation.evaluate_test(changed, run)
    actual = read_json(output / "metrics.json")
    assert actual["metrics"] == expected
    assert actual["metrics"]["false_negative_cost"] == 500
    assert actual["threshold_source"] == "validation"
    np.testing.assert_array_equal(
        pd.read_csv(output / "predictions.csv")["prediction"],
        (positive_scores(pipeline, features.iloc[:30]) >= metadata["threshold"]).astype(int),
    )
    with pytest.raises(ValueError, match="already has"):
        evaluation.evaluate_test(changed, run)


def test_final_evaluation_rejects_a_changed_source_snapshot(
    tmp_path, classification_frame, monkeypatch
):
    features, target = classification_frame
    config = Config(raw_dir=tmp_path / "raw", runs_dir=tmp_path / "runs")
    run = run_experiment(features, target, config, source={"files": {TEST_NAME: "original"}})
    monkeypatch.setattr(evaluation, "verify_raw", lambda _: {"files": {TEST_NAME: "changed"}})
    with pytest.raises(ValueError, match="snapshot differs"):
        evaluation.evaluate_test(config, run)


def test_shared_receipt_blocks_a_second_run_before_test_loading(
    tmp_path, classification_frame, monkeypatch
):
    features, target = classification_frame
    manifest = {"files": {TEST_NAME: "fixture-shared"}}
    config = Config(raw_dir=tmp_path / "raw", runs_dir=tmp_path / "runs")
    first = run_experiment(features, target, config, source=manifest)
    second = run_experiment(features, target, config, source=manifest)
    monkeypatch.setattr(evaluation, "verify_raw", lambda _: manifest)
    calls = []

    def loader(path):
        calls.append(path)
        return features.iloc[:30], target.iloc[:30]

    monkeypatch.setattr(evaluation, "load_aps", loader)
    evaluation.evaluate_test(config, first)
    with pytest.raises(ValueError, match="already has a final-test evaluation attempt"):
        evaluation.evaluate_test(config, second)
    assert len(calls) == 1
    assert not (second / "final_test").exists()


def test_interrupted_test_keeps_consumed_receipt(tmp_path, classification_frame, monkeypatch):
    features, target = classification_frame
    manifest = {"files": {TEST_NAME: "fixture-interrupted"}}
    config = Config(raw_dir=tmp_path / "raw", runs_dir=tmp_path / "runs")
    first = run_experiment(features, target, config, source=manifest)
    second = run_experiment(features, target, config, source=manifest)
    monkeypatch.setattr(evaluation, "verify_raw", lambda _: manifest)

    def interrupt(path):
        raise KeyboardInterrupt("fixture")

    monkeypatch.setattr(evaluation, "load_aps", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_test(config, first)
    assert (
        read_json(next((tmp_path / "final_evaluations").glob("*/attempt.json")))["status"]
        == "failed"
    )
    with pytest.raises(ValueError, match="already has a final-test evaluation attempt"):
        evaluation.evaluate_test(config, second)

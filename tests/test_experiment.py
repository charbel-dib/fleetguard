import json
from pathlib import Path

import pandas as pd
import pytest

from fleetguard.cli import main
from fleetguard.config import Config
from fleetguard.evaluate import evaluate_test
from fleetguard.experiment import require_complete, run_experiment
from fleetguard.io import read_json


def test_experiment_batch_inference_and_no_test_access(tmp_path, classification_frame):
    features, target = classification_frame
    # raw_dir does not exist: no official test data can be read in this experiment.
    config = Config(raw_dir=tmp_path / "nonexistent", runs_dir=tmp_path / "runs")
    run = run_experiment(
        features, target, config, source={"dataset": "fixture"}, kind="synthetic_smoke"
    )
    assert require_complete(run)["status"] == "complete"
    assert not (run / "final_test").exists()
    results = pd.read_csv(run / "validation_metrics.csv")
    assert len(results) == 8
    assert set(results["model"]) == {
        "always_negative",
        "class_prior",
        "logistic",
        "logistic_balanced",
    }
    for name in results["model"].unique():
        metadata = read_json(run / name / "metadata.json")
        selected = metadata["validation"]["cost"]
        default = metadata["validation_at_0_5"]["cost"]
        assert selected <= default
    input_file, output_file = tmp_path / "input.csv", tmp_path / "output.csv"
    features.iloc[:4].to_csv(input_file, index=False, na_rep="na")
    assert (
        main(
            ["predict", "--run", str(run), "--input", str(input_file), "--output", str(output_file)]
        )
        == 0
    )
    prediction = pd.read_csv(output_file)
    assert len(prediction) == 4
    assert prediction["positive_score"].between(0, 1).all()
    assert (
        main(
            ["predict", "--run", str(run), "--input", str(input_file), "--output", str(output_file)]
        )
        == 1
    )
    with pytest.raises(ValueError, match="official APS"):
        evaluate_test(config, run)


def test_incomplete_runs_are_not_usable(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="complete training"):
        require_complete(tmp_path)


def test_invalid_configuration_fails_early():
    with pytest.raises(ValueError):
        Config(raw_dir=Path("raw"), runs_dir=Path("runs"), n_splits=1)
    with pytest.raises(ValueError):
        Config(raw_dir=Path("raw"), runs_dir=Path("runs"), C=0)

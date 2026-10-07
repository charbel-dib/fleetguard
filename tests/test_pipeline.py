import numpy as np
import pandas as pd
import pytest

from fleetguard.artifacts import load_model, positive_scores, prepare_input, save_model
from fleetguard.config import Config
from fleetguard.models import make_models


def test_preprocessing_only_learns_from_training_rows(tmp_path):
    training = pd.DataFrame({"a": [0.0, 2.0, np.nan, 4.0], "empty": [np.nan] * 4})
    target = pd.Series([0, 0, 1, 1])
    pipeline = make_models(Config(raw_dir=tmp_path, runs_dir=tmp_path))["logistic"]
    pipeline.fit(training, target)
    before = pipeline.named_steps["imputer"].statistics_.copy()
    scaler_before = pipeline.named_steps["scaler"].mean_.copy()
    pipeline.predict_proba(pd.DataFrame({"a": [1_000_000.0, np.nan], "empty": [10.0, np.nan]}))
    np.testing.assert_array_equal(before, pipeline.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(scaler_before, pipeline.named_steps["scaler"].mean_)
    assert before[0] == 2.0
    assert np.isfinite(pipeline.named_steps["imputer"].transform(training)).all()


def test_artifact_roundtrip_preserves_scores(tmp_path, classification_frame):
    features, target = classification_frame
    pipeline = make_models(Config(raw_dir=tmp_path, runs_dir=tmp_path))["logistic"]
    pipeline.fit(features, target)
    directory = tmp_path / "model"
    save_model(
        directory,
        pipeline,
        metadata={"feature_names": features.columns.tolist(), "threshold": 0.12},
    )
    reloaded, metadata = load_model(directory)
    ordered = prepare_input(
        features.loc[:, list(reversed(features.columns))], metadata["feature_names"]
    )
    np.testing.assert_allclose(
        positive_scores(pipeline, features), positive_scores(reloaded, ordered)
    )


def test_corrupted_artifact_is_rejected(tmp_path, classification_frame):
    features, target = classification_frame
    pipeline = make_models(Config(raw_dir=tmp_path, runs_dir=tmp_path))["class_prior"]
    pipeline.fit(features, target)
    directory = tmp_path / "model"
    save_model(directory, pipeline, metadata={"threshold": 0.5})
    with (directory / "pipeline.joblib").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model(directory)


@pytest.mark.parametrize("columns", [["a"], ["a", "b", "class"]])
def test_inference_rejects_missing_or_extra_features(columns):
    frame = pd.DataFrame([[1.0] * len(columns)], columns=columns)
    with pytest.raises(ValueError, match="schema mismatch"):
        prepare_input(frame, ["a", "b"])

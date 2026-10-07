from dataclasses import replace

import numpy as np
import pytest

from fleetguard.artifacts import load_model, positive_scores, save_model
from fleetguard.comparison_config import ComparisonConfig
from fleetguard.comparison_models import calibrate_frozen, make_candidates, signed_log1p
from fleetguard.comparison_smoke import smoke_config
from fleetguard.comparison_splits import role_split


def test_signed_log_is_finite_for_large_values_and_preserves_missingness():
    transformed = signed_log1p(np.array([-1e12, -1.0, 0.0, 1.0, 1e12, np.nan]))
    assert np.isfinite(transformed[:5]).all()
    assert np.isnan(transformed[-1])
    assert transformed[0] == -transformed[4]
    assert transformed[2] == 0


def test_factories_use_expected_preprocessing_and_no_internal_holdout(tmp_path):
    models = make_candidates(smoke_config(tmp_path))
    assert not models["logistic_no_indicator"]["imputer"].add_indicator
    assert models["logistic"]["imputer"].add_indicator
    assert "signed_log" in models["logistic_log_robust"].named_steps
    assert models["hist_gradient_boosting"]["classifier"].early_stopping is False
    assert models["xgboost"]["classifier"].get_params()["device"] == "cpu"


def test_sigmoid_calibration_keeps_the_base_estimator_frozen(tmp_path, classification_frame):
    features, target = classification_frame
    roles, _ = role_split(
        features, target, n_splits=5, seed=43, calibration_fold=0, threshold_fold=1
    )
    estimator = make_candidates(smoke_config(tmp_path))["logistic"]
    estimator.fit(features.iloc[roles["fit"]], target.iloc[roles["fit"]])
    before = estimator["classifier"].coef_.copy()
    median_before = estimator["imputer"].statistics_.copy()
    calibrated = calibrate_frozen(
        estimator, features.iloc[roles["calibration"]], target.iloc[roles["calibration"]]
    )
    np.testing.assert_array_equal(before, estimator["classifier"].coef_)
    np.testing.assert_array_equal(median_before, estimator["imputer"].statistics_)
    scores = positive_scores(calibrated, features.iloc[roles["threshold"]])
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_xgboost_artifact_roundtrip_and_version_guard(tmp_path, classification_frame):
    import xgboost

    features, target = classification_frame
    estimator = make_candidates(smoke_config(tmp_path))["xgboost"]
    estimator.fit(features, target)
    metadata = {"threshold": 0.12, "xgboost_version": xgboost.__version__}
    directory = tmp_path / "artifact"
    save_model(directory, estimator, metadata=metadata)
    reloaded, _ = load_model(directory)
    np.testing.assert_allclose(
        positive_scores(estimator, features), positive_scores(reloaded, features)
    )
    monkey_version = xgboost.__version__
    try:
        xgboost.__version__ = "mismatch"
        with pytest.raises(ValueError, match="XGBoost versions differ"):
            load_model(directory)
    finally:
        xgboost.__version__ = monkey_version


def test_invalid_candidate_lists_are_rejected(tmp_path):
    config = smoke_config(tmp_path)
    with pytest.raises(ValueError, match="Unknown models"):
        replace(config, models=("unknown",))
    with pytest.raises(ValueError, match="without duplicates"):
        replace(config, models=("logistic", "logistic"))
    assert isinstance(config, ComparisonConfig)

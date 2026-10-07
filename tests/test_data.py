import numpy as np
import pandas as pd
import pytest

from fleetguard.data import load_aps, profile, validate_frame


def test_csv_preamble_and_na_are_handled(tmp_path):
    path = tmp_path / "fixture.csv"
    path.write_text("Dataset copyright\n\nclass,aa_000,ab_000\nneg,1,na\npos,2,3\n")
    features, target = load_aps(path, official=False)
    assert features.columns.tolist() == ["aa_000", "ab_000"]
    assert np.isnan(features.iloc[0, 1])
    assert target.tolist() == [0, 1]


@pytest.mark.parametrize("value", ["broken", "", "inf", "-inf"])
def test_invalid_sensor_values_fail(value):
    frame = pd.DataFrame({"class": ["neg", "pos"], "sensor": [value, "2"]})
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_unknown_labels_fail():
    frame = pd.DataFrame({"class": ["neg", "positive"], "sensor": ["1", "2"]})
    with pytest.raises(ValueError, match="unknown labels"):
        validate_frame(frame)


def test_duplicate_column_names_are_rejected_before_pandas_mangles_them(tmp_path):
    path = tmp_path / "fixture.csv"
    path.write_text("class,sensor,sensor\nneg,1,2\npos,2,3\n")
    with pytest.raises(ValueError, match="duplicate column"):
        load_aps(path, official=False)


def test_official_loader_rejects_a_tiny_fixture(tmp_path):
    path = tmp_path / "aps_failure_training_set.csv"
    columns = ["class"] + [f"aa_{i:03d}" for i in range(170)]
    path.write_text(
        ",".join(columns)
        + "\n"
        + "neg,"
        + ",".join(["1"] * 170)
        + "\n"
        + "pos,"
        + ",".join(["2"] * 170)
        + "\n"
    )
    with pytest.raises(ValueError, match="60000 rows"):
        load_aps(path)


def test_profile_reports_duplicates_and_conflicting_targets():
    features = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [np.nan] * 3})
    result = profile(features, pd.Series([0, 1, 0]))
    assert result["duplicate_feature_rows"] == 1
    assert result["duplicate_groups_with_conflicting_labels"] == 1
    assert result["all_missing_features"] == ["b"]

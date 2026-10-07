import numpy as np
import pytest

from fleetguard.diagnostics import error_by_missingness, reliability_table


def test_reliability_bins_include_zero_one_and_all_observations():
    table = reliability_table([0, 0, 1, 1], [0.0, 0.05, 0.5, 1.0], bins=10)
    assert table["count"].sum() == 4
    assert table["bin"].tolist() == [0, 5, 9]
    assert table.iloc[0]["mean_score"] == 0.025
    assert table.iloc[-1]["positive_rate"] == 1.0


def test_error_segments_partition_rows_and_preserve_total_cost():
    costs = {"false_positive_cost": 10, "false_negative_cost": 500}
    errors = error_by_missingness(
        [0, 1, 0, 1], [0.9, 0.1, 0.9, 0.1], [0.0, 0.05, 0.2, 1.0], threshold=0.5, costs=costs
    )
    assert errors["rows"].sum() == 4
    assert errors["cost"].sum() == 1020
    assert errors["fp"].sum() == 2
    assert errors["fn"].sum() == 2


def test_invalid_missing_fractions_are_rejected():
    with pytest.raises(ValueError):
        error_by_missingness(
            [0, 1],
            [0.1, 0.8],
            [0.1, np.nan],
            threshold=0.5,
            costs={"false_positive_cost": 10, "false_negative_cost": 500},
        )

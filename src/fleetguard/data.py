"""Load the original APS CSVs without depending on a fixed preamble length."""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_NAME = "aps_failure_training_set.csv"
TEST_NAME = "aps_failure_test_set.csv"
DESCRIPTION_NAME = "aps_failure_description.txt"
EXPECTED_FEATURES = 170
EXPECTED_ROWS = {TRAIN_NAME: 60_000, TEST_NAME: 16_000}
LABELS = {"neg": 0, "pos": 1}


def _header_line(path: Path) -> tuple[int, list[str]]:
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream):
            if line_number > 200:
                break
            columns = next(csv.reader([line]))
            if columns and columns[0] == "class":
                if len(set(columns)) != len(columns):
                    raise ValueError(f"{path.name}: duplicate column names.")
                return line_number, columns
    raise ValueError(f"{path.name}: no APS CSV header found in the first 201 lines.")


def load_aps(path: Path, *, official: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    header_line, columns = _header_line(path)
    if official:
        if len(columns) != EXPECTED_FEATURES + 1:
            raise ValueError(f"{path.name}: expected class + {EXPECTED_FEATURES} sensor columns.")
        if any(
            re.fullmatch(r"[a-z]{2}_[0-9]{3}", c) is None and c not in {"am_0", "ec_00"}
            for c in columns[1:]
        ):
            raise ValueError(f"{path.name}: unexpected sensor column names.")
    frame = pd.read_csv(
        path, skiprows=header_line, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    return validate_frame(frame, official=official, filename=path.name)


def validate_frame(
    frame: pd.DataFrame, *, official: bool = False, filename: str = "dataset"
) -> tuple[pd.DataFrame, pd.Series]:
    if frame.empty or "class" not in frame or frame.columns.duplicated().any():
        raise ValueError(f"{filename}: empty data, missing class or duplicate columns.")
    unexpected = set(frame["class"].unique()) - set(LABELS)
    if unexpected:
        raise ValueError(f"{filename}: unknown labels {sorted(map(str, unexpected))}.")
    if set(frame["class"].unique()) != set(LABELS):
        raise ValueError(f"{filename}: both neg and pos labels are required.")
    if official and len(frame) != EXPECTED_ROWS.get(filename):
        raise ValueError(
            f"{filename}: expected {EXPECTED_ROWS.get(filename)} rows, got {len(frame)}."
        )
    if official and filename == TRAIN_NAME and int((frame["class"] == "pos").sum()) != 1000:
        raise ValueError("Official training set must contain exactly 1,000 positive examples.")
    features = frame.drop(columns="class").replace("na", np.nan)
    if (features == "").to_numpy().any():
        raise ValueError(f"{filename}: empty sensor strings; expected numeric data or 'na'.")
    if features.shape[1] == 0:
        raise ValueError(f"{filename}: no features.")
    try:
        features = features.apply(pd.to_numeric, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{filename}: a feature contains a non-numeric value other than 'na'."
        ) from exc
    if np.isinf(features.to_numpy()).any():
        raise ValueError(f"{filename}: infinite sensor values are not accepted.")
    target = frame["class"].map(LABELS).astype("int8").rename("target")
    return features, target


def profile(features: pd.DataFrame, target: pd.Series) -> dict:
    row_hashes = pd.util.hash_pandas_object(features, index=False)
    conflicts = (
        pd.DataFrame({"group": row_hashes, "target": target}).groupby("group")["target"].nunique()
    )
    return {
        "rows": len(features),
        "features": features.shape[1],
        "negative": int((target == 0).sum()),
        "positive": int((target == 1).sum()),
        "positive_rate": float(target.mean()),
        "missing_fraction": float(features.isna().to_numpy().mean()),
        "missing_by_feature": {k: float(v) for k, v in features.isna().mean().items()},
        "all_missing_features": features.columns[features.isna().all()].tolist(),
        "constant_observed_features": features.columns[features.nunique() <= 1].tolist(),
        "duplicate_feature_rows": int(row_hashes.duplicated().sum()),
        "duplicate_groups_with_conflicting_labels": int((conflicts > 1).sum()),
    }

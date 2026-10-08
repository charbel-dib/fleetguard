"""Persist and reload complete inference pipelines, with a versioned contract."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from fleetguard.io import read_json, sha256_file, write_json

SCHEMA_VERSION = 1


def save_model(directory: Path, pipeline, *, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "pipeline.joblib"
    joblib.dump(pipeline, path, compress=3)
    write_json(
        directory / "metadata.json",
        {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "sklearn_version": sklearn.__version__,
            "pipeline_sha256": sha256_file(path),
        },
    )


def load_model(directory: Path) -> tuple[object, dict]:
    metadata = read_json(directory / "metadata.json")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported model artifact schema.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError(
            "Model/runtime scikit-learn versions differ; use the recorded environment."
        )
    if metadata.get("xgboost_version"):
        import xgboost

        if xgboost.__version__ != metadata["xgboost_version"]:
            raise ValueError("Model/runtime XGBoost versions differ; use the recorded environment.")
    if metadata.get("torch_version"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("This MLP artifact requires FleetGuard's research extra.") from exc

        if torch.__version__ != metadata["torch_version"]:
            raise ValueError("Model/runtime PyTorch versions differ; use the recorded environment.")
    path = directory / "pipeline.joblib"
    if sha256_file(path) != metadata["pipeline_sha256"]:
        raise ValueError("Model checksum mismatch.")
    # joblib/pickle can execute code: only load artifacts created by a trusted local run.
    return joblib.load(path), metadata


def prepare_input(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Prediction input must be nonempty and have unique column names.")
    missing = set(feature_names) - set(frame.columns)
    extra = set(frame.columns) - set(feature_names)
    if missing or extra:
        raise ValueError(
            f"Input schema mismatch. Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )
    try:
        values = frame.loc[:, feature_names].replace("na", np.nan)
        values = values.apply(pd.to_numeric, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError("Prediction features must be numeric or missing ('na'/empty).") from exc
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Prediction features must not contain infinities.")
    return values


def positive_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError("Pipeline has no unambiguous positive class (1).")
    return pipeline.predict_proba(features)[:, indices[0]]

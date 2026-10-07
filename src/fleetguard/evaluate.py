"""Explicit final-test evaluation using frozen models and validation thresholds."""

from pathlib import Path

import pandas as pd

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.config import Config
from fleetguard.data import TEST_NAME, load_aps
from fleetguard.download import verify_raw
from fleetguard.experiment import require_complete
from fleetguard.io import read_json, write_json
from fleetguard.metrics import classification_metrics


def evaluate_test(config: Config, run_dir: Path) -> Path:
    run = require_complete(run_dir)
    if run["kind"] != "official_aps":
        raise ValueError("Final-test evaluation requires a run trained on official APS data.")
    manifest = verify_raw(config.raw_dir)
    source = read_json(run_dir / "source.json")
    if manifest["files"] != source["files"]:
        raise ValueError("Data snapshot differs from the training run.")
    output = run_dir / "final_test"
    if output.exists():
        raise ValueError(
            "This run already has a final-test evaluation; it will not be overwritten."
        )
    features, target = load_aps(config.raw_dir / TEST_NAME)
    model_name = read_json(run_dir / "champion.json")["model"]
    pipeline, metadata = load_model(run_dir / model_name)
    features = prepare_input(features, metadata["feature_names"])
    scores = positive_scores(pipeline, features)
    metrics = classification_metrics(
        target, scores, threshold=metadata["threshold"], **metadata["costs"]
    )
    # Report exact feature overlap without changing the official test or tuning on it.
    assignments = pd.read_csv(run_dir / "split.csv", dtype={"feature_group": "uint64"})
    test_groups = pd.util.hash_pandas_object(features, index=False)
    overlap = int(test_groups.isin(assignments["feature_group"]).sum())
    output.mkdir()
    write_json(
        output / "metrics.json",
        {
            "model": model_name,
            "metrics": metrics,
            "feature_rows_also_present_in_official_train": overlap,
            "test_file_sha256": manifest["files"][TEST_NAME],
            "threshold_source": "validation",
        },
    )
    pd.DataFrame(
        {
            "row_id": range(len(features)),
            "target": target,
            "score": scores,
            "prediction": (scores >= metadata["threshold"]).astype(int),
        }
    ).to_csv(output / "predictions.csv", index=False)
    return output

"""One-shot final-test evaluation with a shared local receipt and frozen decisions."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.config import Config
from fleetguard.data import TEST_NAME, load_aps
from fleetguard.download import verify_raw
from fleetguard.experiment import require_complete
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.metrics import classification_metrics
from fleetguard.uncertainty import grouped_intervals

logger = logging.getLogger("fleetguard")


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
    model_name = read_json(run_dir / "champion.json")["model"]
    pipeline, metadata = load_model(run_dir / model_name)
    # Pin all selection inputs before loading the test. This is a local guard,
    # shared even when a different run directory is passed to the command.
    test_hash = manifest["files"][TEST_NAME]
    key = hashlib.sha256(test_hash.encode()).hexdigest()
    receipt = config.raw_dir.resolve().parent / "final_evaluations" / key
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            "This data snapshot already has a final-test evaluation attempt. "
            "Shared receipt prevents scoring another run or retrying an interrupted attempt."
        ) from exc
    attempt = {
        "status": "started",
        "run_id": run_dir.name,
        "model": model_name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "test_file_sha256": test_hash,
        "pipeline_sha256": metadata["pipeline_sha256"],
        "threshold": metadata["threshold"],
        "costs": metadata["costs"],
        "freeze_sha256": sha256_file(run_dir / "freeze.json")
        if run.get("protocol") == "release_v1"
        else None,
    }
    write_json(receipt / "attempt.json", attempt)
    try:
        output.mkdir()
        features, target = load_aps(config.raw_dir / TEST_NAME)
        features = prepare_input(features, metadata["feature_names"])
        with threadpool_limits(limits=1):
            scores = positive_scores(pipeline, features)
        metrics = classification_metrics(
            target, scores, threshold=metadata["threshold"], **metadata["costs"]
        )
        assignments = pd.read_csv(run_dir / "split.csv", dtype={"feature_group": "uint64"})
        test_groups = pd.util.hash_pandas_object(features, index=False)
        overlap = int(test_groups.isin(assignments["feature_group"]).sum())
        threshold_source = metadata.get(
            "threshold_source",
            "development_threshold_role"
            if run.get("protocol") in {"comparison_v1", "optimization_v1"}
            else "validation",
        )
        result = {
            "model": model_name,
            "metrics": metrics,
            "feature_rows_also_present_in_official_train": overlap,
            "test_file_sha256": test_hash,
            "threshold_source": threshold_source,
            "evaluation_policy": "frozen_champion_only; no test policy sweep or retuning",
            "receipt": str(receipt),
            "freeze_sha256": attempt["freeze_sha256"],
        }
        write_json(output / "metrics.json", result)
        pd.DataFrame(
            {
                "row_id": range(len(features)),
                "target": target,
                "score": scores,
                "prediction": (scores >= metadata["threshold"]).astype(int),
            }
        ).to_csv(output / "predictions.csv", index=False)
        write_json(
            output / "uncertainty.json",
            grouped_intervals(
                target,
                scores,
                test_groups,
                threshold=metadata["threshold"],
                costs=metadata["costs"],
            ),
        )
        write_json(
            output / "status.json",
            {"status": "complete", "metrics_sha256": sha256_file(output / "metrics.json")},
        )
        write_json(
            receipt / "attempt.json",
            {
                **attempt,
                "status": "complete",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "metrics_sha256": sha256_file(output / "metrics.json"),
            },
        )
    except BaseException as exc:
        write_json(
            receipt / "attempt.json",
            {**attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        write_json(output / "status.json", {"status": "failed"})
        raise
    if run.get("protocol") == "release_v1":
        from fleetguard.release_report import write_release_report

        # Report regeneration only reads saved outcomes; it never rescores test.
        write_release_report(run_dir)
    return output

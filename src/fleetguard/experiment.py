"""Run the validation benchmark; the official test is never opened here."""

import importlib.metadata
import platform
import time
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from fleetguard.artifacts import positive_scores, save_model
from fleetguard.config import Config
from fleetguard.data import TRAIN_NAME, load_aps, profile
from fleetguard.download import verify_raw
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.metrics import classification_metrics, select_threshold, threshold_curve
from fleetguard.models import make_models
from fleetguard.splits import make_split


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "fleetguard",
            )
        },
    }


def run_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    config: Config,
    *,
    source: dict,
    kind: str = "official_aps",
) -> Path:
    training, validation, assignments = make_split(
        features,
        target,
        seed=config.seed,
        n_splits=config.n_splits,
        validation_fold=config.validation_fold,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    assignments.to_csv(run_dir / "split.csv", index=False)
    write_json(run_dir / "data_profile.json", profile(features, target))
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "source.json", source)
    write_json(run_dir / "environment.json", _environment())
    write_json(run_dir / "run.json", {"id": run_id, "kind": kind, "status": "running"})
    costs = {
        "false_positive_cost": config.false_positive_cost,
        "false_negative_cost": config.false_negative_cost,
    }
    x_train, y_train = features.iloc[training], target.iloc[training]
    x_validation, y_validation = features.iloc[validation], target.iloc[validation]
    rows, candidate_metadata = [], {}
    try:
        for name, pipeline in make_models(config).items():
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            scores = positive_scores(pipeline, x_validation)
            curve = threshold_curve(y_validation, scores, **costs)
            tuned = select_threshold(curve)
            # The always-negative reference keeps its intended fixed decision rule.
            threshold = 0.5 if name == "always_negative" else tuned
            metrics = classification_metrics(y_validation, scores, threshold=threshold, **costs)
            default_metrics = classification_metrics(y_validation, scores, threshold=0.5, **costs)
            metadata = {
                "model": name,
                "run_id": run_id,
                "kind": kind,
                "feature_names": features.columns.tolist(),
                "threshold": threshold,
                "costs": costs,
                "validation": metrics,
                "validation_at_0_5": default_metrics,
                "score_semantics": "uncalibrated positive-class model score",
                "training_rows": len(training),
                "validation_rows": len(validation),
                "fit_seconds": fit_seconds,
            }
            save_model(run_dir / name, pipeline, metadata=metadata)
            curve.to_csv(run_dir / name / "threshold_curve.csv", index=False)
            pd.DataFrame(
                {
                    "row_id": validation,
                    "target": y_validation.to_numpy(),
                    "score": scores,
                    "prediction": (scores >= threshold).astype(int),
                    "missing_fraction": x_validation.isna().mean(axis=1).to_numpy(),
                }
            ).to_csv(run_dir / name / "validation_predictions.csv", index=False)
            candidate_metadata[name] = metadata
            for rule, result in (("threshold_0.5", default_metrics), ("selected", metrics)):
                rows.append({"model": name, "rule": rule, "fit_seconds": fit_seconds, **result})
        leaderboard = pd.DataFrame(rows)
        leaderboard.to_csv(run_dir / "validation_metrics.csv", index=False)
        selected = leaderboard[leaderboard["rule"] == "selected"].sort_values(
            ["cost", "average_precision", "model"], ascending=[True, False, True]
        )
        champion = str(selected.iloc[0]["model"])
        write_json(
            run_dir / "champion.json",
            {
                "model": champion,
                "selection_partition": "validation",
                "selection_metric": "cost_then_average_precision",
            },
        )
        _write_report(run_dir, candidate_metadata, champion)
        write_json(
            run_dir / "run.json",
            {
                "id": run_id,
                "kind": kind,
                "status": "complete",
                "split_sha256": sha256_file(run_dir / "split.csv"),
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "run.json",
            {"id": run_id, "kind": kind, "status": "failed", "error": str(exc)},
        )
        raise
    return run_dir


def _write_report(run_dir: Path, candidates: dict, champion: str) -> None:
    lines = [
        "# FleetGuard — validation benchmark",
        "",
        f"Run: `{run_dir.name}`. Selected model: `{champion}`.",
        "",
        "Thresholds and model selection use the same validation partition. These are development",
        "results, not unbiased final-test estimates. No official test metrics are computed here.",
        "",
        "| Model | Rule | Threshold | AP | Recall | Precision | FP | FN | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metadata in candidates.items():
        for rule, metrics in (
            ("0.5", metadata["validation_at_0_5"]),
            ("selected", metadata["validation"]),
        ):
            lines.append(
                f"| {name} | {rule} | {metrics['threshold']:.6g} | "
                f"{metrics['average_precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['fp']} | "
                f"{metrics['fn']} | {metrics['cost']} |"
            )
    lines.extend(
        [
            "",
            "Cost units follow the dataset challenge; they are not euros.",
            "Scores from the logistic models have not been calibrated.",
            "",
        ]
    )
    (run_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def train(config: Config) -> Path:
    manifest = verify_raw(config.raw_dir)
    features, target = load_aps(config.raw_dir / TRAIN_NAME)
    return run_experiment(features, target, config, source=manifest)


def require_complete(run_dir: Path) -> dict:
    run = read_json(run_dir / "run.json")
    if run["status"] != "complete":
        raise ValueError("Only a complete training run can be used for inference or evaluation.")
    if sha256_file(run_dir / "split.csv") != run["split_sha256"]:
        raise ValueError("Split manifest checksum mismatch.")
    if run.get("protocol") == "comparison_v1":
        for filename, key in (
            ("cv_roles.csv", "cv_roles_sha256"),
            ("final_roles.csv", "final_roles_sha256"),
        ):
            if sha256_file(run_dir / filename) != run[key]:
                raise ValueError(f"Comparison role manifest checksum mismatch: {filename}.")
    return run

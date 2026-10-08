"""Offline release contract smoke using a simplified optimization artifact fixture."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fleetguard.artifacts import load_model, positive_scores, save_model
from fleetguard.comparison_splits import role_split
from fleetguard.experiment import require_complete
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.metrics import select_threshold, threshold_curve
from fleetguard.release import run_audit
from fleetguard.release_config import ReleaseConfig
from fleetguard.splits import make_split


def main():
    with tempfile.TemporaryDirectory(prefix="fleetguard-release-smoke-") as temporary:
        root = Path(temporary)
        x, y = make_classification(
            n_samples=800, n_features=8, n_informative=5, weights=[0.8, 0.2], random_state=42
        )
        features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
        target = pd.Series(y)
        development, _, split = make_split(features, target, seed=42, n_splits=5, validation_fold=0)
        x_dev, y_dev = features.iloc[development], target.iloc[development]
        roles, assignments = role_split(
            x_dev, y_dev, n_splits=5, seed=43, calibration_fold=0, threshold_fold=1
        )
        parent = root / "source"
        parent.mkdir()
        split.to_csv(parent / "split.csv", index=False)
        assignments["row_id"] = development
        assignments.drop(columns="local_row").to_csv(parent / "final_roles.csv", index=False)
        assignments.to_csv(parent / "cv_roles.csv", index=False)
        model = Pipeline(
            [
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=20,
                        max_depth=2,
                        learning_rate=0.2,
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            ]
        )
        model.fit(x_dev.iloc[roles["fit"]], y_dev.iloc[roles["fit"]])
        scores = positive_scores(model, x_dev.iloc[roles["threshold"]])
        threshold = select_threshold(threshold_curve(y_dev.iloc[roles["threshold"]], scores))
        import xgboost

        save_model(
            parent / "xgboost_tuned",
            model,
            metadata={
                "calibration": None,
                "threshold": threshold,
                "feature_names": features.columns.tolist(),
                "training_rows": len(roles["fit"]),
                "costs": {"false_positive_cost": 10, "false_negative_cost": 500},
                "parameters": {"n_estimators": 20, "max_depth": 2},
                "xgboost_version": xgboost.__version__,
            },
        )
        write_json(parent / "source.json", {"dataset": "synthetic software fixture"})
        write_json(parent / "champion.json", {"model": "xgboost_tuned"})
        write_json(
            parent / "run.json",
            {
                "status": "complete",
                "kind": "synthetic_smoke",
                "protocol": "optimization_v1",
                "split_sha256": sha256_file(parent / "split.csv"),
                "cv_roles_sha256": sha256_file(parent / "cv_roles.csv"),
                "final_roles_sha256": sha256_file(parent / "final_roles.csv"),
            },
        )
        run = run_audit(
            features,
            target,
            ReleaseConfig(
                raw_dir=root / "raw",
                releases_dir=root / "releases",
                calibration_folds=2,
                permutation_repeats=1,
                benchmark_repeats=2,
                batch_sizes=(1, 16),
            ),
            parent,
        )
        require_complete(run)
        model, _ = load_model(run / "xgboost_release")
        if not np.isfinite(positive_scores(model, features.iloc[:3])).all():
            raise RuntimeError("Release inference failed.")
        if (run / "final_test").exists() or read_json(run / "decision.json")[
            "official_test_used_for_selection"
        ]:
            raise RuntimeError("Release audit crossed the final-test boundary.")
    print(
        "Release smoke passed: isolated calibration, frozen decision, diagnostics, "
        "CPU benchmark and artifact reload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

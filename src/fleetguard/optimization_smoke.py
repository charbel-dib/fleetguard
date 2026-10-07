"""Offline research smoke: trial budgets, roles, tracking, MLP and persisted inference."""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fleetguard.artifacts import load_model, positive_scores
from fleetguard.comparison_smoke import smoke_config
from fleetguard.experiment import require_complete
from fleetguard.io import read_json
from fleetguard.optimization_config import OptimizationConfig


def research_smoke_config(root):
    return OptimizationConfig(
        comparison=replace(smoke_config(root), xgb_estimators=80),
        tracking_dir=root / "tracking",
        xgb_trials=1,
        mlp_trials=1,
        mlp_max_epochs=2,
        mlp_patience=1,
        mlp_inner_folds=2,
        stability_seeds=(43,),
        learning_fractions=(1.0,),
    )


def _run_fixture(root):
    from fleetguard.optimization import run_optimization

    x, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.85, 0.15], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    features.iloc[::7, 0] = np.nan
    run = run_optimization(
        features,
        pd.Series(y),
        research_smoke_config(root),
        source={"dataset": "synthetic software fixture"},
        kind="synthetic_smoke",
    )
    require_complete(run)
    champion = read_json(run / "champion.json")["model"]
    model, _ = load_model(run / champion)
    if not np.isfinite(positive_scores(model, features.iloc[:3])).all():
        raise RuntimeError("Optimization artifact inference failed.")
    if len(pd.read_csv(run / "trials.csv")) != 2:
        raise RuntimeError("Trial budget was not respected.")
    if not (root / "tracking/mlflow.db").exists() or not (run / "figures/mlp_epochs.png").exists():
        raise RuntimeError("Tracking or report missing.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root is not None:
        _run_fixture(args.worker_root)
        return 0

    # MLflow caches SQLite connection pools without a public client close API.
    # Keep all database activity in a child; process exit releases MLflow AND
    # Optuna file handles before TemporaryDirectory removes them on Windows.
    # Do not suppress cleanup errors or report success after a worker failure.
    with tempfile.TemporaryDirectory(prefix="fleetguard-optimization-smoke-") as folder:
        worker = subprocess.run(
            [sys.executable, "-m", "fleetguard.optimization_smoke", "--worker-root", folder],
            check=False,
        )
        if worker.returncode:
            return worker.returncode if worker.returncode > 0 else 1
    print(
        "Optimization smoke passed: Optuna, MLflow SQLite, MLP, controls, "
        "role audits and artifact reload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

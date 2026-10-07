"""Exercise every new candidate and the comparison protocol without network access."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fleetguard.artifacts import load_model, positive_scores
from fleetguard.comparison import run_comparison
from fleetguard.comparison_config import ComparisonConfig
from fleetguard.config import Config
from fleetguard.experiment import require_complete
from fleetguard.io import read_json


def smoke_config(root: Path) -> ComparisonConfig:
    return ComparisonConfig(
        base=Config(raw_dir=root / "raw", runs_dir=root / "runs"),
        cv_folds=2,
        forest_estimators=8,
        forest_depth=4,
        hgb_iterations=8,
        hgb_leaves=7,
        xgb_estimators=8,
        xgb_depth=2,
        n_jobs=1,
    )


def main() -> int:
    x, y = make_classification(
        n_samples=600, n_features=12, n_informative=7, weights=[0.85, 0.15], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(12)])
    features.iloc[::7, 0] = np.nan
    with tempfile.TemporaryDirectory(prefix="fleetguard-comparison-smoke-") as temporary:
        root = Path(temporary)
        run = run_comparison(
            features,
            pd.Series(y),
            smoke_config(root),
            source={"dataset": "synthetic software fixture"},
            kind="synthetic_smoke",
        )
        require_complete(run)
        champion = read_json(run / "champion.json")["model"]
        estimator, _ = load_model(run / champion)
        scores = positive_scores(estimator, features.iloc[:3])
        if scores.shape != (3,) or not np.isfinite(scores).all():
            raise RuntimeError("Comparison artifact inference failed.")
        summary = pd.read_csv(run / "cv_summary.csv")
        if len(summary) != 10 or not (run / "figures/cv_cost.png").is_file():
            raise RuntimeError("Comparison outputs incomplete.")
    print(
        "Comparison smoke passed: 10 candidates, isolated roles, scoring folds, "
        "plots and artifact reload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline integration smoke test. Its synthetic scores are not APS benchmark results."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.config import Config
from fleetguard.experiment import require_complete, run_experiment
from fleetguard.io import read_json


def main() -> int:
    x, y = make_classification(
        n_samples=240, n_features=12, n_informative=6, weights=[0.9, 0.1], random_state=42
    )
    frame = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(x.shape[1])])
    frame.iloc[::7, 0] = np.nan
    frame["empty_sensor"] = np.nan
    with tempfile.TemporaryDirectory(prefix="fleetguard-smoke-") as temporary:
        root = Path(temporary)
        config = Config(raw_dir=root / "raw", runs_dir=root / "runs")
        run = run_experiment(
            frame,
            pd.Series(y),
            config,
            source={"dataset": "synthetic integration fixture", "seed": 42},
            kind="synthetic_smoke",
        )
        require_complete(run)
        champion = read_json(run / "champion.json")["model"]
        pipeline, metadata = load_model(run / champion)
        reordered = frame.iloc[:3].loc[:, list(reversed(frame.columns))]
        scores = positive_scores(pipeline, prepare_input(reordered, metadata["feature_names"]))
        if not np.isfinite(scores).all() or scores.shape != (3,):
            raise RuntimeError("Smoke inference failed.")
    print(
        "Offline smoke passed: grouped split, training, metrics, artifact reload, batch inference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

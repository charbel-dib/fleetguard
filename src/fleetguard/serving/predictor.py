"""Load a trusted release once, verify its freeze, and serve deterministic scores."""

import math
import threading

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.experiment import require_complete
from fleetguard.io import read_json, sha256_file

FLOAT32_MAX = float(np.finfo(np.float32).max)


class InputError(ValueError):
    def __init__(self, details):
        super().__init__("Sensor columns must match the frozen model schema.")
        self.details = details


class Predictor:
    def __init__(self, settings):
        directory = settings.release_dir
        run = require_complete(directory)
        if run.get("protocol") != "release_v1":
            raise ValueError("Serving requires an audited release_v1 artifact, not a training run.")
        if run["kind"] != "official_aps" and not (
            run["kind"] == "synthetic_smoke" and settings.allow_synthetic
        ):
            raise ValueError("Synthetic release serving requires explicit fixture mode.")
        freeze = read_json(directory / "freeze.json")
        decision = read_json(directory / "decision.json")
        if decision["policy"] != "challenge_cost":
            raise ValueError("Serving v1 supports only the frozen challenge_cost policy.")
        name = freeze["model"]
        model, metadata = load_model(directory / name)
        if (
            metadata["calibration"] != decision["calibration"]
            or metadata["costs"] != decision["costs"]
        ):
            raise ValueError("Decision differs from the model calibration/cost metadata.")
        features = metadata["feature_names"]
        if run["kind"] == "official_aps" and len(features) != 170:
            raise ValueError("Official APS releases require 170 sensor columns.")
        if (
            not features
            or len(set(features)) != len(features)
            or not all(isinstance(x, str) for x in features)
        ):
            raise ValueError("Release feature schema is invalid.")
        if metadata.get("release_id") != freeze["release_id"]:
            raise ValueError("Release identity differs between freeze and model metadata.")
        threshold = metadata["threshold"]
        if not math.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, np.inf):
            raise ValueError("Release threshold is invalid.")
        self._model = model
        self._lock = threading.Lock()
        self.feature_names = tuple(features)
        self._feature_set = set(features)
        self.max_batch_rows = settings.max_batch_rows
        self.identity = {
            "release_id": freeze["release_id"],
            "name": name,
            "dataset_kind": run["kind"],
            "pipeline_sha256": metadata["pipeline_sha256"],
            "freeze_sha256": sha256_file(directory / "freeze.json"),
            "threshold": float(threshold),
            "calibration": metadata["calibration"],
            "policy": "challenge_cost",
            "score_semantics": metadata["score_semantics"],
        }
        self.costs = dict(metadata["costs"])
        # Validate deserialization and the inference path before becoming ready.
        self.predict([{name: None for name in features}])

    def predict(self, rows):
        if not 1 <= len(rows) <= self.max_batch_rows:
            raise InputError(
                [{"loc": ["body", "rows"], "message": "Batch row limit exceeded or empty batch."}]
            )
        details = []
        for index, row in enumerate(rows):
            missing, extra = self._feature_set - set(row), set(row) - self._feature_set
            if missing or extra:
                details.append(
                    {"row_index": index, "missing": sorted(missing), "extra": sorted(extra)}
                )
            for name, value in row.items():
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or abs(value) > FLOAT32_MAX
                ):
                    raise InputError(
                        [
                            {
                                "row_index": index,
                                "feature": name,
                                "message": "Use a finite float32-compatible number or null.",
                            }
                        ]
                    )
        if details:
            raise InputError(details[:20])
        frame = prepare_input(pd.DataFrame(rows), list(self.feature_names))
        # Serialize inference: threadpoolctl changes process-level pool settings.
        # Async API work runs in a dedicated thread, so health checks remain responsive.
        with self._lock, threadpool_limits(limits=1):
            scores = positive_scores(self._model, frame)
        if (
            scores.shape != (len(rows),)
            or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise RuntimeError("Model produced invalid scores.")
        return [
            {
                "row_index": index,
                "positive_score": float(score),
                "predicted_label": "pos" if score >= self.identity["threshold"] else "neg",
                "missing_fraction": float(frame.iloc[index].isna().mean()),
            }
            for index, score in enumerate(scores)
        ]

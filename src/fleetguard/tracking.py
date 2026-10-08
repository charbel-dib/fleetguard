"""Explicit local SQLite tracking; never inherit an external MLflow destination."""

import math
import os
import time
from pathlib import Path


class LocalTracker:
    def __init__(self, directory: Path):
        # Set before importing MLflow; the tracking scope is strictly local.
        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
        from mlflow import MlflowClient
        from mlflow.telemetry import get_telemetry_client

        if get_telemetry_client() is not None:
            raise RuntimeError("MLflow telemetry must be disabled for local tracking.")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.uri = "sqlite:///" + (self.directory / "mlflow.db").as_posix()
        self.client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        name = "FleetGuard-optimization-v1"
        experiment = self.client.get_experiment_by_name(name)
        self.experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment(
                name, artifact_location=(self.directory / "mlflow-artifacts").as_uri()
            )
        )

    def start(self, name, *, parent=None, tags=None):
        values = {"mlflow.runName": name, "protocol": "optimization_v1", **(tags or {})}
        if parent:
            values["mlflow.parentRunId"] = parent
        return self.client.create_run(self.experiment_id, tags=values).info.run_id

    def log(self, run_id, *, params=None, metrics=None, tags=None):
        from mlflow.entities import Metric, Param, RunTag

        self.client.log_batch(
            run_id,
            params=[Param(str(k), str(v)) for k, v in (params or {}).items()],
            metrics=[
                Metric(str(k), float(v), int(time.time() * 1000), 0)
                for k, v in (metrics or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            ],
            tags=[RunTag(str(k), str(v)) for k, v in (tags or {}).items()],
        )

    def artifact(self, run_id, path):
        self.client.log_artifact(run_id, str(path))

    def finish(self, run_id, *, failed=False):
        self.client.set_terminated(run_id, status="FAILED" if failed else "FINISHED")

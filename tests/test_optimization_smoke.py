import os
import subprocess
import sys
from importlib.util import find_spec

import pytest


def test_research_smoke_releases_database_files_before_cleanup(tmp_path):
    for package in ("torch", "optuna", "imblearn", "mlflow"):
        # Check availability without importing MLflow in the parent process.
        if find_spec(package) is None:
            pytest.skip(f"Research dependency {package} is unavailable.")

    env = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "MLFLOW_DISABLE_AGENT_HINT": "true",
    }
    result = subprocess.run(
        [sys.executable, "-m", "fleetguard.optimization_smoke"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Optimization smoke passed:" in result.stdout
    assert not list(tmp_path.glob("fleetguard-optimization-smoke-*"))

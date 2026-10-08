"""Content checks for a model/threshold frozen before final-test access."""

from pathlib import Path

from fleetguard.io import read_json, sha256_file, write_json


def freeze_release(directory, *, model):
    paths = [
        "source.json",
        "config.json",
        "split.csv",
        "final_roles.csv",
        "calibration_roles.csv",
        "champion.json",
        "decision.json",
        "policy_thresholds.csv",
        f"{model}/pipeline.joblib",
        f"{model}/metadata.json",
    ]
    decision = read_json(directory / "decision.json")
    value = {
        "schema_version": 1,
        "release_id": directory.name,
        "model": model,
        "threshold": decision["threshold"],
        "frozen_at_utc": decision["frozen_at_utc"],
        "threshold_source": "development_threshold_role",
        "official_test_used_for_selection": False,
        "files": {path: sha256_file(directory / path) for path in paths},
    }
    write_json(directory / "freeze.json", value)
    return value


def verify_release(directory):
    freeze = read_json(directory / "freeze.json")
    if freeze["schema_version"] != 1 or freeze["official_test_used_for_selection"] is not False:
        raise ValueError("Unsupported release freeze.")
    model = read_json(directory / "champion.json")["model"]
    required = {
        "source.json",
        "config.json",
        "split.csv",
        "final_roles.csv",
        "calibration_roles.csv",
        "champion.json",
        "decision.json",
        "policy_thresholds.csv",
        f"{model}/pipeline.joblib",
        f"{model}/metadata.json",
    }
    if set(freeze["files"]) != required:
        raise ValueError("Release freeze file set mismatch.")
    for relative, expected in freeze["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha256_file(directory / path) != expected:
            raise ValueError(f"Release freeze checksum mismatch: {relative}.")
    metadata = read_json(directory / model / "metadata.json")
    decision = read_json(directory / "decision.json")
    if not (
        freeze["model"] == model
        and freeze["threshold"] == metadata["threshold"] == decision["threshold"]
        and freeze["threshold_source"] == "development_threshold_role"
        and freeze["frozen_at_utc"] == decision["frozen_at_utc"]
    ):
        raise ValueError("Release model/threshold freeze mismatch.")
    return freeze

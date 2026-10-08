"""Validate a build candidate without accepting tags, mutable models or arbitrary registries."""

import math
import re

from fleetguard.cloud.artifact import require_sha256
from fleetguard.serving.schemas import ModelIdentity

IMAGE = re.compile(
    r"(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z]{2}-[a-z]+-\d)\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9][a-z0-9._/-]*)@sha256:(?P<digest>[0-9a-f]{64})\Z"
)


def validate_candidate(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "git_sha",
        "build_run_id",
        "bundle_sha256",
        "image",
        "model",
    }:
        raise ValueError("Invalid cloud candidate fields.")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("Unsupported cloud candidate schema.")
    if not isinstance(value["git_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["git_sha"]):
        raise ValueError("Candidate must identify a full Git commit.")
    if not isinstance(value["build_run_id"], str) or not re.fullmatch(
        r"[1-9][0-9]*", value["build_run_id"]
    ):
        raise ValueError("Candidate must identify its successful GitHub build run.")
    require_sha256(value["bundle_sha256"])
    if not isinstance(value["image"], str) or not IMAGE.fullmatch(value["image"]):
        raise ValueError("Candidate requires a private ECR image pinned by SHA256 digest.")
    model = ModelIdentity.model_validate(value["model"]).model_dump()
    if model["dataset_kind"] != "official_aps" or model != value["model"]:
        raise ValueError("Only an official, exact model identity can be promoted.")
    require_sha256(model["pipeline_sha256"])
    require_sha256(model["freeze_sha256"])
    if not math.isfinite(model["threshold"]) or not 0 <= model["threshold"] <= 1.0000000000000002:
        raise ValueError("Invalid frozen threshold.")
    if not model["release_id"] or not model["name"] or not model["score_semantics"]:
        raise ValueError("Incomplete model identity.")
    return value


def require_staging_receipt(receipt, candidate):
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("status") == "passed"
        and receipt.get("environment") == "staging"
        and receipt.get("candidate") == candidate
        and receipt.get("checks", {}).get("inference_contract") is True
        and receipt.get("checks", {}).get("deployment_identity") is True
        and receipt.get("checks", {}).get("web_assets") is True
    ):
        raise ValueError(
            "Production requires a successful staging receipt for this exact candidate."
        )

"""Deploy one exact ECS image, verify it, and restore the previous task on failure."""

import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from fleetguard.cloud.candidate import IMAGE, validate_candidate
from fleetguard.cloud.probe import probe_candidate
from fleetguard.io import write_json
from fleetguard.serving.probe import request_json

REGISTER_FIELDS = {
    "family",
    "taskRoleArn",
    "executionRoleArn",
    "networkMode",
    "containerDefinitions",
    "volumes",
    "placementConstraints",
    "requiresCompatibilities",
    "cpu",
    "memory",
    "pidMode",
    "ipcMode",
    "proxyConfiguration",
    "inferenceAccelerators",
    "ephemeralStorage",
    "runtimePlatform",
    "enableFaultInjection",
}


class RolloutFailed(RuntimeError):
    """ECS reports failure or rollback of the requested revision."""


class AwsCli:
    def __init__(self, region):
        self.region = region

    def call(self, *arguments):
        result = subprocess.run(
            ["aws", *arguments, "--region", self.region, "--output", "json", "--no-cli-pager"],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout or "{}")

    def register(self, definition):
        with tempfile.TemporaryDirectory(prefix="fleetguard-task-") as folder:
            path = Path(folder) / "task.json"
            write_json(path, definition)
            return self.call(
                "ecs", "register-task-definition", "--cli-input-json", f"file://{path}"
            )


def service_state(aws, cluster, service):
    value = aws.call("ecs", "describe-services", "--cluster", cluster, "--services", service)
    if value.get("failures") or len(value.get("services", [])) != 1:
        raise RuntimeError("ECS service was not found.")
    return value["services"][0]


def wait_for_revision(aws, cluster, service, task_arn, *, timeout=600, pause=time.sleep):
    started = time.monotonic()
    deadline = started + timeout
    requested_seen = False
    while True:
        state = service_state(aws, cluster, service)
        primary = next((x for x in state["deployments"] if x["status"] == "PRIMARY"), None)
        if primary is None or primary["taskDefinition"] != task_arn:
            # DescribeServices can briefly show the old PRIMARY after UpdateService.
            if requested_seen or time.monotonic() - started >= min(60, timeout):
                raise RolloutFailed(
                    "ECS replaced the requested deployment, possibly by circuit-breaker rollback."
                )
            pause(5)
            continue
        requested_seen = True
        if primary.get("rolloutState") == "FAILED":
            raise RolloutFailed("ECS marked the deployment as failed.")
        if (
            primary.get("rolloutState") == "COMPLETED"
            and len(state["deployments"]) == 1
            and state["runningCount"] == state["desiredCount"] > 0
            and state["pendingCount"] == 0
        ):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("ECS deployment deadline exceeded.")
        pause(5)


def revised_definition(source, candidate, *, drill=False):
    definition = {key: value for key, value in source.items() if key in REGISTER_FIELDS}
    # Copy before changing nested fields, so the rollback source remains immutable.
    definition = json.loads(json.dumps(definition))
    containers = definition["containerDefinitions"]
    if len(containers) != 1 or containers[0]["name"] != "fleetguard":
        raise ValueError(
            "Expected the single FleetGuard container from the runtime infrastructure."
        )
    container = containers[0]
    container["image"] = candidate["image"]
    selected = {
        "FLEETGUARD_GIT_SHA": candidate["git_sha"],
        "FLEETGUARD_BUNDLE_SHA256": candidate["bundle_sha256"],
        "FLEETGUARD_IMAGE_DIGEST": candidate["image"].split("@", 1)[1],
        "FLEETGUARD_ALLOW_SYNTHETIC": "false",
        "FLEETGUARD_RELEASE_DIR": "/models/missing-rollback-drill" if drill else "/models/release",
    }
    environment = {item["name"]: item["value"] for item in container.get("environment", [])}
    environment.update(selected)
    container["environment"] = [
        {"name": name, "value": value} for name, value in sorted(environment.items())
    ]
    return definition


def deploy_candidate(
    candidate,
    *,
    aws,
    cluster,
    service,
    repository,
    url,
    environment,
    report_path,
    drill=False,
    check=probe_candidate,
    wait=wait_for_revision,
):
    validate_candidate(candidate)
    if environment not in {"staging", "production"} or service != f"{repository}-{environment}":
        raise ValueError("Environment/service does not match the selected FleetGuard repository.")
    if drill and environment != "staging":
        raise ValueError("Failure drills are restricted to staging.")
    image = IMAGE.fullmatch(candidate["image"])
    account = aws.call("sts", "get-caller-identity")["Account"]
    if (
        image["account"] != account
        or image["region"] != aws.region
        or image["repository"] != repository
    ):
        raise ValueError("Candidate registry differs from the selected account/region/repository.")
    aws.call(
        "ecr",
        "describe-images",
        "--repository-name",
        repository,
        "--image-ids",
        f"imageDigest=sha256:{image['digest']}",
    )
    state = service_state(aws, cluster, service)
    previous_arn = state["taskDefinition"]
    wait(aws, cluster, service, previous_arn)
    source = aws.call("ecs", "describe-task-definition", "--task-definition", previous_arn)[
        "taskDefinition"
    ]
    if source["family"] != service:
        raise ValueError("Task definition family differs from the infrastructure service.")
    current = source["containerDefinitions"][0]
    current_env = {item["name"]: item["value"] for item in current.get("environment", [])}
    _, info, _ = request_json(url.rstrip("/") + "/v1/model")
    previous = {
        "schema_version": 1,
        "git_sha": current_env["FLEETGUARD_GIT_SHA"],
        "build_run_id": "1",
        "bundle_sha256": current_env["FLEETGUARD_BUNDLE_SHA256"],
        "image": current["image"],
        "model": info["model"],
    }
    validate_candidate(previous)
    check(url, previous)
    if drill and previous["image"] != candidate["image"]:
        raise ValueError("A rollback drill must target the current candidate.")
    report = {
        "schema_version": 1,
        "status": "failed",
        "environment": environment,
        "candidate": candidate,
        "url": url,
        "previous_task_definition": previous_arn,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "rollback_verified": False,
    }
    updated = False
    try:
        new_arn = aws.register(revised_definition(source, candidate, drill=drill))[
            "taskDefinition"
        ]["taskDefinitionArn"]
        report["requested_task_definition"] = new_arn
        # Set before the request: a lost response may still have updated ECS.
        updated = True
        aws.call(
            "ecs",
            "update-service",
            "--cluster",
            cluster,
            "--service",
            service,
            "--task-definition",
            new_arn,
        )
        wait(aws, cluster, service, new_arn)
        checks = check(url, candidate)
        if drill:
            raise RuntimeError("The intentionally missing model unexpectedly became healthy.")
        report.update(status="passed", checks=checks)
    except Exception as error:
        report["failure_type"] = type(error).__name__
        if updated:
            try:
                aws.call(
                    "ecs",
                    "update-service",
                    "--cluster",
                    cluster,
                    "--service",
                    service,
                    "--task-definition",
                    previous_arn,
                )
                wait(aws, cluster, service, previous_arn)
                report["rollback_checks"] = check(url, previous)
                report["rollback_verified"] = True
            except Exception as rollback_error:
                report["rollback_failure_type"] = type(rollback_error).__name__
        if drill and report["rollback_verified"] and "checks" not in report:
            # Only an ECS rollout failure proves the intentionally broken startup failed.
            if report.get("requested_task_definition") and isinstance(error, RolloutFailed):
                report["status"] = "rollback_verified"
            else:
                raise
        else:
            raise
    finally:
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(Path(report_path), report)
    return report

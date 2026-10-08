import copy

import pytest

pytest.importorskip("pydantic")
from fleetguard.cloud.candidate import require_staging_receipt, validate_candidate  # noqa: E402
from fleetguard.cloud.deploy import RolloutFailed, deploy_candidate, wait_for_revision  # noqa: E402
from fleetguard.io import read_json  # noqa: E402


def candidate(digest="a"):
    return {
        "schema_version": 1,
        "git_sha": "b" * 40,
        "build_run_id": "123",
        "bundle_sha256": "c" * 64,
        "image": "123456789012.dkr.ecr.eu-west-3.amazonaws.com/fleetguard@sha256:" + digest * 64,
        "model": {
            "release_id": "fixture-identity-only",
            "name": "xgboost_release",
            "dataset_kind": "official_aps",
            "pipeline_sha256": "d" * 64,
            "freeze_sha256": "e" * 64,
            "threshold": 0.05,
            "calibration": "raw",
            "policy": "challenge_cost",
            "score_semantics": "identity-only test; no Scania model",
        },
    }


CHECKS = {"inference_contract": True, "deployment_identity": True, "web_assets": True}


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "repo:latest"),
        ("git_sha", "short"),
        ("bundle_sha256", "wrong"),
        ("build_run_id", "123;whoami"),
        ("schema_version", True),
    ],
)
def test_invalid_candidate_is_rejected(field, value):
    value_to_test = candidate()
    value_to_test[field] = value
    with pytest.raises(ValueError):
        validate_candidate(value_to_test)


def test_production_requires_staging_of_the_exact_digest_model_and_code():
    value = candidate()
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "environment": "staging",
        "candidate": value,
        "checks": CHECKS,
    }
    require_staging_receipt(receipt, value)
    with pytest.raises(ValueError):
        require_staging_receipt(receipt, candidate("f"))
    with pytest.raises(ValueError):
        require_staging_receipt({**receipt, "status": "rollback_verified"}, value)


class FakeAws:
    region = "eu-west-3"

    def __init__(self):
        old = candidate("f")
        self.updates = []
        self.source = {
            "family": "fleetguard-staging",
            "taskDefinitionArn": "old",
            "revision": 1,
            "containerDefinitions": [
                {
                    "name": "fleetguard",
                    "image": old["image"],
                    "readonlyRootFilesystem": True,
                    "environment": [
                        {"name": "FLEETGUARD_GIT_SHA", "value": old["git_sha"]},
                        {"name": "FLEETGUARD_BUNDLE_SHA256", "value": old["bundle_sha256"]},
                    ],
                }
            ],
        }

    def call(self, *arguments):
        if arguments[0] == "sts":
            return {"Account": "123456789012"}
        if arguments[:2] == ("ecr", "describe-images"):
            return {}
        if arguments[1] == "describe-services":
            return {"services": [{"taskDefinition": "old"}]}
        if arguments[1] == "describe-task-definition":
            return {"taskDefinition": self.source}
        if arguments[1] == "update-service":
            self.updates.append(arguments[-1])
            return {}
        raise AssertionError(arguments)

    def register(self, definition):
        self.registered = definition
        return {"taskDefinition": {"taskDefinitionArn": "new"}}


def perform(monkeypatch, tmp_path, *, failure=None, rollback_failure=False, drill=False):
    aws = FakeAws()
    expected = candidate("f" if drill else "a")
    monkeypatch.setattr(
        "fleetguard.cloud.deploy.request_json", lambda _: (200, {"model": expected["model"]}, {})
    )
    original = copy.deepcopy(aws.source)
    checks = []

    def check(url, value):
        checks.append(value["image"])
        if value["image"] == expected["image"] and len(checks) > 1 and failure == "probe":
            raise ValueError("Wrong image/model served after rollout")
        if rollback_failure and len(checks) > 1:
            raise OSError("Rollback endpoint unavailable")
        return CHECKS

    def wait(aws, cluster, service, revision):
        if revision == "new" and failure == "rollout":
            raise RolloutFailed("ECS marked the deployment as failed.")

    kwargs = dict(
        aws=aws,
        cluster="fleetguard-staging",
        service="fleetguard-staging",
        repository="fleetguard",
        url="https://example.test",
        environment="staging",
        report_path=tmp_path / "report.json",
        drill=drill,
        check=check,
        wait=wait,
    )
    if failure and not drill:
        with pytest.raises((ValueError, RolloutFailed)):
            deploy_candidate(expected, **kwargs)
    else:
        deploy_candidate(expected, **kwargs)
    assert aws.source == original
    return aws, read_json(tmp_path / "report.json")


def test_verified_deploy_changes_only_revision_and_preserves_runtime_config(monkeypatch, tmp_path):
    aws, report = perform(monkeypatch, tmp_path)
    assert report["status"] == "passed" and aws.updates == ["new"]
    assert aws.registered["containerDefinitions"][0]["readonlyRootFilesystem"] is True
    assert "taskDefinitionArn" not in aws.registered


@pytest.mark.parametrize("failure", ["probe", "rollout"])
def test_failure_restores_previous_revision_and_verifies_previous_identity(
    monkeypatch, tmp_path, failure
):
    aws, report = perform(monkeypatch, tmp_path, failure=failure)
    assert aws.updates == ["new", "old"]
    assert report["status"] == "failed" and report["rollback_verified"] is True


def test_failed_rollback_cannot_report_success(monkeypatch, tmp_path):
    _, report = perform(monkeypatch, tmp_path, failure="rollout", rollback_failure=True)
    assert report["rollback_verified"] is False and report["rollback_failure_type"] == "OSError"


def test_staging_drill_requires_rollout_failure_and_verifies_restored_service(
    monkeypatch, tmp_path
):
    aws, report = perform(monkeypatch, tmp_path, failure="rollout", drill=True)
    env = {x["name"]: x["value"] for x in aws.registered["containerDefinitions"][0]["environment"]}
    assert env["FLEETGUARD_RELEASE_DIR"] == "/models/missing-rollback-drill"
    assert report["status"] == "rollback_verified" and report["rollback_verified"] is True


def test_ecs_stable_after_automatic_rollback_is_not_a_successful_new_deployment():
    class RolledBack:
        def call(self, *args):
            return {
                "services": [
                    {
                        "deployments": [
                            {
                                "status": "PRIMARY",
                                "taskDefinition": "old",
                                "rolloutState": "COMPLETED",
                            }
                        ]
                    }
                ]
            }

    with pytest.raises(RolloutFailed):
        wait_for_revision(
            RolledBack(), "cluster", "service", "new", timeout=0, pause=lambda _: None
        )


def test_waiter_tolerates_brief_old_primary_after_update():
    class EventuallyConsistent:
        calls = 0

        def call(self, *args):
            self.calls += 1
            primary = "old" if self.calls == 1 else "new"
            return {
                "services": [
                    {
                        "deployments": [
                            {
                                "status": "PRIMARY",
                                "taskDefinition": primary,
                                "rolloutState": "COMPLETED",
                            }
                        ],
                        "runningCount": 1,
                        "desiredCount": 1,
                        "pendingCount": 0,
                    }
                ]
            }

    aws = EventuallyConsistent()
    wait_for_revision(aws, "cluster", "service", "new", pause=lambda _: None)
    assert aws.calls == 2

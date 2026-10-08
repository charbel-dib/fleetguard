"""Check GitHub provenance and download only successful main-branch receipts."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlencode

from fleetguard.cloud.candidate import require_staging_receipt, validate_candidate
from fleetguard.io import read_json


def gh(*arguments):
    return subprocess.check_output(["gh", *arguments], text=True)


def successful_run(repository, run_id, workflow):
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("Invalid GitHub run ID.")
    result = json.loads(gh("api", f"repos/{repository}/actions/runs/{run_id}"))
    if not (
        result["status"] == "completed"
        and result["conclusion"] == "success"
        and result["head_branch"] == "main"
        and result["event"] == "workflow_dispatch"
        and result["path"] == f".github/workflows/{workflow}"
        and result["repository"]["full_name"] == repository
    ):
        raise ValueError(
            "Receipt must originate from the expected successful main-branch workflow."
        )
    return result


def download(repository, run_id, artifact, destination, filename):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use a fresh receipt destination.")
    gh(
        "run",
        "download",
        run_id,
        "--repo",
        repository,
        "--name",
        artifact,
        "--dir",
        str(destination),
    )
    files = [p for p in destination.rglob("*") if p.is_file()]
    if files != [destination / filename]:
        raise ValueError("Unexpected receipt artifact contents.")
    return read_json(destination / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-ci", "candidate", "staging"])
    parser.add_argument("--run-id")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if os.environ["GITHUB_REF"] != "refs/heads/main":
        raise ValueError("Cloud workflows must be dispatched from main.")
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.command == "check-ci":
        query = urlencode(
            {"head_sha": os.environ["GITHUB_SHA"], "status": "completed", "per_page": 100}
        )
        result = json.loads(gh("api", f"repos/{repository}/actions/workflows/ci.yml/runs?{query}"))
        eligible = [
            r
            for r in result["workflow_runs"]
            if r["head_branch"] == "main" and r["event"] in {"push", "workflow_dispatch"}
        ]
        latest = max(eligible, key=lambda r: (r["run_number"], r["run_attempt"]), default=None)
        if latest is None or latest["conclusion"] != "success":
            raise ValueError(
                "The latest completed main CI for this exact commit must be successful."
            )
        print(f"Verified successful CI run {latest['id']} for this commit.")
    elif args.command == "candidate":
        run = successful_run(repository, args.run_id, "publish-cloud.yml")
        candidate = validate_candidate(
            download(repository, args.run_id, "cloud-candidate", args.destination, "candidate.json")
        )
        if candidate["build_run_id"] != args.run_id or candidate["git_sha"] != run["head_sha"]:
            raise ValueError("Candidate differs from its GitHub build provenance.")
        print("Verified immutable build candidate.")
    else:
        successful_run(repository, args.run_id, "deploy-cloud.yml")
        receipt = download(
            repository, args.run_id, "cloud-deployment-receipt", args.destination, "deployment.json"
        )
        require_staging_receipt(receipt, validate_candidate(read_json(args.candidate)))
        print("Verified successful staging receipt for this exact candidate.")


if __name__ == "__main__":
    main()

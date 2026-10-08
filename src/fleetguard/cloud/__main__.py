"""Cloud helpers: python -m fleetguard.cloud --help."""

import argparse
import json
from pathlib import Path

from fleetguard.cloud.artifact import pack_model, unpack_model
from fleetguard.cloud.candidate import require_staging_receipt, validate_candidate
from fleetguard.cloud.deploy import AwsCli, deploy_candidate
from fleetguard.cloud.probe import probe_candidate
from fleetguard.io import read_json, write_json
from fleetguard.serving.predictor import Predictor
from fleetguard.serving.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--release", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--bundle", type=Path, required=True)
    candidate.add_argument("--sha256", required=True)
    candidate.add_argument("--image", required=True)
    candidate.add_argument("--git-sha", required=True)
    candidate.add_argument("--run-id", required=True)
    candidate.add_argument("--output", type=Path, required=True)
    inputs = commands.add_parser("terraform-input")
    inputs.add_argument("--candidate", type=Path, required=True)
    inputs.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--candidate", type=Path, required=True)
    probe.add_argument("--url", required=True)
    deploy = commands.add_parser("deploy")
    for name in ("candidate", "report", "staging-receipt"):
        deploy.add_argument(f"--{name}", type=Path, required=name != "staging-receipt")
    for name in ("cluster", "service", "repository", "region", "url"):
        deploy.add_argument(f"--{name}", required=True)
    deploy.add_argument("--environment", choices=["staging", "production"], required=True)
    deploy.add_argument("--rollback-drill", action="store_true")
    args = parser.parse_args()
    if args.command == "pack":
        result = pack_model(args.release, args.output)
    elif args.command == "unpack":
        result = unpack_model(args.archive, args.sha256, args.destination)
    elif args.command == "candidate":
        model = Predictor(Settings(release_dir=args.bundle)).identity
        result = validate_candidate(
            {
                "schema_version": 1,
                "git_sha": args.git_sha,
                "build_run_id": args.run_id,
                "bundle_sha256": args.sha256,
                "image": args.image,
                "model": model,
            }
        )
        write_json(args.output, result)
    else:
        value = validate_candidate(read_json(args.candidate))
        if args.command == "terraform-input":
            result = {"candidate": value}
            write_json(args.output, result)
        elif args.command == "probe":
            result = probe_candidate(args.url, value)
        else:
            if args.environment == "production":
                if args.staging_receipt is None:
                    parser.error("Production requires --staging-receipt.")
                require_staging_receipt(read_json(args.staging_receipt), value)
            result = deploy_candidate(
                value,
                aws=AwsCli(args.region),
                cluster=args.cluster,
                service=args.service,
                repository=args.repository,
                url=args.url,
                environment=args.environment,
                report_path=args.report,
                drill=args.rollback_drill,
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

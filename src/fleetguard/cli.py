"""Commands are deliberately separated at the official test boundary."""

import argparse
import json
import logging
import urllib.error
import zipfile
from pathlib import Path

import pandas as pd

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.config import load_config
from fleetguard.data import TRAIN_NAME, load_aps, profile
from fleetguard.download import download, verify_raw
from fleetguard.evaluate import evaluate_test
from fleetguard.experiment import require_complete, train
from fleetguard.io import read_json, write_json

logger = logging.getLogger("fleetguard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FleetGuard APS classification benchmark.")
    parser.add_argument("--verbose", action="store_true", help="Show traceback on command failure.")
    commands = parser.add_subparsers(dest="command", required=True)
    download_parser = commands.add_parser(
        "download", help="Download or import the official UCI ZIP."
    )
    download_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    download_parser.add_argument(
        "--archive", type=Path, help="Import an already-downloaded UCI ZIP."
    )
    validate = commands.add_parser(
        "validate", help="Verify source hashes and profile the training data."
    )
    validate.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    validate.add_argument("--output", type=Path, default=Path("reports/data_quality.json"))
    training = commands.add_parser(
        "train", help="Fit baselines; select thresholds using validation only."
    )
    training.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    comparison = commands.add_parser(
        "compare", help="Compare models using train-only CV and frozen thresholds."
    )
    comparison.add_argument("--config", type=Path, default=Path("configs/comparison.toml"))
    evaluation = commands.add_parser(
        "evaluate-test", help="Evaluate the frozen champion on official test."
    )
    evaluation.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument(
        "--final",
        action="store_true",
        required=True,
        help="Explicitly request the final-test evaluation.",
    )
    prediction = commands.add_parser(
        "predict", help="Batch inference from a plain sensor-only CSV."
    )
    prediction.add_argument("--run", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "download":
        manifest = download(args.raw_dir, archive=args.archive)
        logger.info("Registered UCI snapshot: %s", manifest["archive_sha256"])
    elif args.command == "validate":
        config = load_config(args.config)
        verify_raw(config.raw_dir)
        features, target = load_aps(config.raw_dir / TRAIN_NAME)
        result = profile(features, target)
        write_json(args.output, result)
        logger.info(
            "Validated %s rows, %s features, %s positives. Report: %s",
            result["rows"],
            result["features"],
            result["positive"],
            args.output,
        )
    elif args.command == "train":
        run = train(load_config(args.config))
        logger.info("Run complete: %s", run)
        print((run / "validation_report.md").read_text(encoding="utf-8"))
    elif args.command == "evaluate-test":
        output = evaluate_test(load_config(args.config), args.run)
        logger.info("Final-test results: %s", output)
        print(json.dumps(read_json(output / "metrics.json"), indent=2))
    elif args.command == "compare":
        from fleetguard.comparison import compare
        from fleetguard.comparison_config import load_comparison_config

        run = compare(load_comparison_config(args.config))
        logger.info("Comparison complete: %s", run)
        print((run / "comparison_report.md").read_text(encoding="utf-8"))
    elif args.command == "predict":
        require_complete(args.run)
        if args.output.exists():
            raise ValueError("Prediction output already exists; choose a new filename.")
        champion = read_json(args.run / "champion.json")["model"]
        pipeline, metadata = load_model(args.run / champion)
        frame = pd.read_csv(args.input, keep_default_na=False, na_values=["na", ""])
        features = prepare_input(frame, metadata["feature_names"])
        scores = positive_scores(pipeline, features)
        output = pd.DataFrame(
            {
                "row_id": range(len(features)),
                "positive_score": scores,
                "predicted_label": ["pos" if p >= metadata["threshold"] else "neg" for p in scores],
                "threshold": metadata["threshold"],
                "model": champion,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        logger.info("Predicted %s rows: %s", len(output), args.output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        _dispatch(args)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        logger.error("%s", exc, exc_info=args.verbose)
        return 1
    return 0

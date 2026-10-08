"""Verify a local service/container using an optional trusted artifact as reference."""

import argparse
import json
from pathlib import Path

from fleetguard.serving.predictor import Predictor
from fleetguard.serving.probe import probe
from fleetguard.serving.settings import Settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--release", type=Path)
parser.add_argument("--expect-synthetic", action="store_true")
args = parser.parse_args()
reference = None
if args.release:
    reference = Predictor(Settings(release_dir=args.release, allow_synthetic=args.expect_synthetic))
result = probe(args.url, expected_predictor=reference, expect_synthetic=args.expect_synthetic)
print(json.dumps(result, indent=2))

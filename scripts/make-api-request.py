"""Build a strict batch JSON request from sensor-only CSV, or an explicit null probe."""

import argparse
from pathlib import Path

import pandas as pd

from fleetguard.artifacts import prepare_input
from fleetguard.io import write_json
from fleetguard.serving.predictor import Predictor
from fleetguard.serving.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Plain sensor-only CSV; never reads UCI labels.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings.from_env(args.release)
    if not 1 <= args.rows <= settings.max_batch_rows:
        parser.error(f"--rows must be in [1, {settings.max_batch_rows}].")
    if args.output.exists():
        parser.error("Output already exists; select a new path.")
    reference = Predictor(settings)
    if args.input:
        frame = pd.read_csv(
            args.input, nrows=args.rows, keep_default_na=False, na_values=["na", ""]
        )
        frame = prepare_input(frame, list(reference.feature_names))
        rows = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    else:
        rows = [{name: None for name in reference.feature_names}]
        print("All-null contract probe only; not a meaningful truck diagnosis.")
    reference.predict(rows)  # Validate schema/range before writing; do not save scores.
    write_json(args.output, {"rows": rows})
    print(f"Created {len(rows)} row(s) at {args.output}; input values stay local.")


if __name__ == "__main__":
    main()

"""Exercise the real loopback HTTP API, or prepare an explicit container fixture."""

import argparse
import json
import socket
import tempfile
import threading
from pathlib import Path

import uvicorn

from fleetguard.artifacts import prepare_input
from fleetguard.io import write_json
from fleetguard.release_smoke import create_fixture
from fleetguard.serving.api import create_app
from fleetguard.serving.bundle import bundle_release
from fleetguard.serving.predictor import Predictor
from fleetguard.serving.probe import probe
from fleetguard.serving.settings import Settings


def run_http_smoke(settings, rows=None):
    expected = Predictor(settings)
    app = create_app(settings)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, access_log=False, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        return probe(
            f"http://127.0.0.1:{port}",
            expected_predictor=expected,
            rows=rows,
            expect_synthetic=settings.allow_synthetic,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("API smoke server did not shut down.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", type=Path, help="Verify an existing official release without training."
    )
    parser.add_argument(
        "--input", type=Path, help="Optional sensor-only CSV; probe its first four rows."
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fixture-out", type=Path, help="Create a synthetic minimal bundle for container CI."
    )
    args = parser.parse_args(argv)
    if args.fixture_out and (args.release or args.input):
        parser.error("Fixture preparation cannot be combined with a real release/input.")
    if args.input and not args.release:
        parser.error("--input requires --release.")
    with tempfile.TemporaryDirectory(prefix="fleetguard-api-smoke-") as folder:
        if args.release:
            settings = Settings(release_dir=args.release.resolve())
            rows = None
            if args.input:
                import pandas as pd

                predictor = Predictor(settings)
                frame = pd.read_csv(args.input, keep_default_na=False, na_values=["na", ""])
                frame = prepare_input(frame.iloc[:4], list(predictor.feature_names))
                rows = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
        else:
            _, _, release = create_fixture(Path(folder))
            if args.fixture_out:
                bundle_release(release, args.fixture_out.resolve())
                print("Synthetic container fixture created (not Scania performance evidence).")
                return 0
            settings = Settings(release_dir=release, allow_synthetic=True)
            rows = None
        result = run_http_smoke(settings, rows)
        if args.report:
            if args.report.exists():
                raise ValueError("Probe report already exists; select a new output.")
            write_json(args.report, result)
    print(
        "API smoke passed: real HTTP, liveness/readiness, identity, "
        "single/batch and local score parity."
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

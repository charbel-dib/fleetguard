"""Serve a built UI with an explicitly synthetic API fixture for browser integration tests."""

import argparse
import tempfile
from pathlib import Path

import uvicorn

from fleetguard.release_smoke import create_fixture
from fleetguard.serving.api import create_app
from fleetguard.serving.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18006)
    parser.add_argument("--web-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fleetguard-web-fixture-") as folder:
        _, _, release = create_fixture(Path(folder))
        settings = Settings(
            release_dir=release,
            web_dir=args.web_dir.resolve(),
            max_batch_rows=2,
            allow_synthetic=True,
        )
        uvicorn.run(
            create_app(settings),
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            server_header=False,
            limit_concurrency=16,
        )


if __name__ == "__main__":
    main()

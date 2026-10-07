"""Fetch the official UCI archive, keeping source files and SHA-256 provenance."""

import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fleetguard.data import DESCRIPTION_NAME, TEST_NAME, TRAIN_NAME, load_aps
from fleetguard.io import read_json, sha256_file, write_json

SOURCE_URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks"
NAMES = (TRAIN_NAME, TEST_NAME, DESCRIPTION_NAME)
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 100 * 1024 * 1024


def verify_raw(raw_dir: Path) -> dict:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Data manifest missing. Run 'fleetguard download' or import the UCI ZIP.")
    manifest = read_json(manifest_path)
    for name in NAMES:
        path = raw_dir / name
        if not path.is_file() or sha256_file(path) != manifest.get("files", {}).get(name):
            raise ValueError(f"Source file missing or modified: {name}. Reimport the official ZIP.")
    return manifest


def _unpack(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for name in NAMES:
            matches = [m for m in zipped.infolist() if Path(m.filename).name == name]
            if len(matches) != 1:
                raise ValueError(f"UCI archive must contain exactly one {name}.")
            member = matches[0]
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Oversized archive member: {name}.")
            # Copy only allowlisted basenames; never extract arbitrary archive paths.
            with zipped.open(member) as source, (destination / name).open("wb") as target:
                shutil.copyfileobj(source, target)


def download(raw_dir: Path, *, archive: Path | None = None) -> dict:
    if (raw_dir / "manifest.json").exists():
        if archive is not None:
            raise ValueError(
                "Data already registered. Import into a new raw directory to change it."
            )
        return verify_raw(raw_dir)
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise ValueError("Raw directory is nonempty without a manifest; use a new directory.")
    with tempfile.TemporaryDirectory(prefix="fleetguard-", dir=raw_dir.parent) as temporary:
        staging = Path(temporary)
        local_archive = staging / "source.zip"
        if archive is None:
            request = urllib.request.Request(
                SOURCE_URL, headers={"User-Agent": "FleetGuard/0.1 (research portfolio project)"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as source,
                local_archive.open("wb") as target,
            ):
                total = 0
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("UCI archive exceeded the 100 MiB download limit.")
                    target.write(block)
        else:
            if archive.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("UCI archive exceeded the 100 MiB size limit.")
            shutil.copyfile(archive, local_archive)
        _unpack(local_archive, staging)
        train_features, _ = load_aps(staging / TRAIN_NAME)
        test_features, _ = load_aps(staging / TEST_NAME)
        if list(train_features.columns) != list(test_features.columns):
            raise ValueError("Official training and test feature schemas differ.")
        manifest = {
            "source_url": SOURCE_URL,
            "source_page": SOURCE_PAGE,
            "doi": "10.24432/C51S51",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "archive_sha256": sha256_file(local_archive),
            "files": {name: sha256_file(staging / name) for name in NAMES},
            "integrity_note": "Hashes identify this local snapshot, not a publisher signature.",
        }
        raw_dir.mkdir(exist_ok=True)
        for name in NAMES:
            shutil.move(staging / name, raw_dir / name)
        write_json(raw_dir / "manifest.json", manifest)
    return manifest

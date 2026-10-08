"""Bounded model archives, validated before any trusted joblib deserialization."""

import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from fleetguard.experiment import require_complete
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.serving.predictor import Predictor
from fleetguard.serving.settings import Settings

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require_sha256(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Expected a lowercase SHA256 digest.")
    return value


def file_names(directory):
    require_complete(directory)
    freeze = read_json(directory / "freeze.json")
    return sorted([*freeze["files"], "freeze.json", "run.json"])


def pack_model(release, output, *, allow_synthetic=False):
    release, output = Path(release), Path(output)
    names = file_names(release)
    predictor = Predictor(Settings(release_dir=release, allow_synthetic=allow_synthetic))
    if sum((release / name).stat().st_size for name in names) > MAX_ARCHIVE_BYTES:
        raise ValueError("Uncompressed model archive exceeds 200 MiB.")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Archive or receipt already exists; choose a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        # Fixed timestamps/order make the same frozen bytes produce the same archive.
        with zipfile.ZipFile(output, "x", zipfile.ZIP_STORED) as archive:
            created = True
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, (release / name).read_bytes())
        if output.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Model archive exceeds 200 MiB.")
        receipt = {
            "schema_version": 1,
            "bundle_sha256": sha256_file(output),
            "s3_key": f"models/{sha256_file(output)}.zip",
            "model": predictor.identity,
            "files": {name: sha256_file(release / name) for name in names},
        }
        write_json(output.with_suffix(".json"), receipt)
        return receipt
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def unpack_model(archive_path, expected_sha256, destination, *, allow_synthetic=False):
    archive_path, destination = Path(archive_path), Path(destination)
    require_sha256(expected_sha256)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Model archive exceeds 200 MiB.")
    if sha256_file(archive_path) != expected_sha256:
        raise ValueError("Model archive SHA256 mismatch.")
    if destination.exists():
        raise ValueError("Destination must not exist.")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != 12 or len(set(names)) != len(names):
            raise ValueError("Expected exactly twelve distinct release files.")
        if sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("Uncompressed model archive exceeds 200 MiB.")
        for item in members:
            name = item.filename
            path = PurePosixPath(name)
            mode = item.external_attr >> 16
            if (
                item.is_dir()
                or "\\" in name
                or ":" in name
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
            ):
                raise ValueError("Unsafe model archive member.")
        freeze = json.loads(archive.read("freeze.json"))
        if set(names) != {*freeze["files"], "freeze.json", "run.json"}:
            raise ValueError("Archive differs from its freeze file set.")
        for name, expected in freeze["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != require_sha256(expected):
                raise ValueError("Archived release file checksum mismatch.")
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for item in members:
                path = destination / item.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))
                path.chmod(0o644)
            # Checks all structural freeze invariants before loading the trusted model.
            require_complete(destination)
            return Predictor(
                Settings(release_dir=destination, allow_synthetic=allow_synthetic)
            ).identity
        except BaseException:
            shutil.rmtree(destination)
            raise

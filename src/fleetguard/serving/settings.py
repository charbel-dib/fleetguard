"""Explicit local release and bounded serving resources."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    release_dir: Path
    max_batch_rows: int = 256
    max_request_bytes: int = 2 * 1024 * 1024
    max_concurrency: int = 16
    allow_synthetic: bool = False

    def __post_init__(self):
        for key, upper in (
            ("max_batch_rows", 1024),
            ("max_request_bytes", 16 * 1024 * 1024),
            ("max_concurrency", 64),
        ):
            value = getattr(self, key)
            # Uvicorn counts the active connection before checking this cap.
            lower = 2 if key == "max_concurrency" else 1
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{key} must be an integer in [{lower}, {upper}].")
        if type(self.allow_synthetic) is not bool:
            raise ValueError("allow_synthetic must be a boolean.")

    @classmethod
    def from_env(cls, release_dir=None):
        selected = release_dir or os.environ.get("FLEETGUARD_RELEASE_DIR")
        if not selected:
            raise ValueError(
                "Set FLEETGUARD_RELEASE_DIR or pass --release with a trusted frozen release."
            )
        test_mode = os.environ.get("FLEETGUARD_ALLOW_SYNTHETIC", "false").lower()
        if test_mode not in {"true", "false"}:
            raise ValueError("FLEETGUARD_ALLOW_SYNTHETIC must be true or false.")
        return cls(
            release_dir=Path(selected).resolve(),
            max_batch_rows=int(os.environ.get("FLEETGUARD_MAX_BATCH_ROWS", "256")),
            max_request_bytes=int(
                os.environ.get("FLEETGUARD_MAX_REQUEST_BYTES", str(2 * 1024 * 1024))
            ),
            max_concurrency=int(os.environ.get("FLEETGUARD_MAX_CONCURRENCY", "16")),
            allow_synthetic=test_mode == "true",
        )

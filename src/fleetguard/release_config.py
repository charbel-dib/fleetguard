"""Fixed audit budget and decision rules; no official test configuration here."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseConfig:
    raw_dir: Path
    releases_dir: Path
    seed: int = 46
    calibration_folds: int = 3
    permutation_repeats: int = 3
    benchmark_repeats: int = 20
    batch_sizes: tuple[int, ...] = (1, 128, 1024)
    inspection_budgets: tuple[float, ...] = (0.01, 0.03, 0.05)

    def __post_init__(self):
        counts = (self.calibration_folds, self.permutation_repeats, self.benchmark_repeats)
        if any(type(n) is not int or n < 1 for n in counts) or self.calibration_folds < 2:
            raise ValueError(
                "Positive integer budgets and at least two calibration folds required."
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        if not self.batch_sizes or any(type(n) is not int or n < 1 for n in self.batch_sizes):
            raise ValueError("Batch sizes must be positive integers.")
        if not self.inspection_budgets or any(
            not math.isfinite(n) or not 0 < n <= 1 for n in self.inspection_budgets
        ):
            raise ValueError("Inspection budgets must be finite fractions in (0, 1].")
        if len(set(self.batch_sizes)) != len(self.batch_sizes) or len(
            set(self.inspection_budgets)
        ) != len(self.inspection_budgets):
            raise ValueError("Duplicate audit budgets are not allowed.")

    def to_dict(self):
        return {
            **asdict(self),
            "raw_dir": str(self.raw_dir),
            "releases_dir": str(self.releases_dir),
        }


def load_release_config(path):
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    if set(values) != {"data", "output", "audit"}:
        raise ValueError("Release config requires data, output and audit sections.")
    if set(values["data"]) != {"raw_dir"} or set(values["output"]) != {"releases_dir"}:
        raise ValueError("Unexpected release paths.")
    if set(values["audit"]) != {
        "seed",
        "calibration_folds",
        "permutation_repeats",
        "benchmark_repeats",
        "batch_sizes",
        "inspection_budgets",
    }:
        raise ValueError("Unexpected audit settings.")
    root = path.resolve().parent.parent
    audit = dict(values["audit"])
    for key in ("batch_sizes", "inspection_budgets"):
        audit[key] = tuple(audit[key])
    return ReleaseConfig(
        raw_dir=(root / values["data"]["raw_dir"]).resolve(),
        releases_dir=(root / values["output"]["releases_dir"]).resolve(),
        **audit,
    )

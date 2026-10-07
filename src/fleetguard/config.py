"""Experiment configuration; relative paths are resolved from the repository root."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    runs_dir: Path
    seed: int = 42
    n_splits: int = 5
    validation_fold: int = 0
    C: float = 1.0
    max_iter: int = 2000
    tol: float = 1e-4
    false_positive_cost: int = 10
    false_negative_cost: int = 500

    def __post_init__(self) -> None:
        if self.n_splits < 2 or not 0 <= self.validation_fold < self.n_splits:
            raise ValueError("Invalid split: n_splits >= 2 and 0 <= validation_fold < n_splits.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        if not math.isfinite(self.C) or self.C <= 0 or self.max_iter < 1:
            raise ValueError("C and max_iter must be positive.")
        if not math.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive.")
        if self.false_positive_cost < 1 or self.false_negative_cost < 1:
            raise ValueError("Both error costs must be positive integers.")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["raw_dir"] = str(self.raw_dir)
        values["runs_dir"] = str(self.runs_dir)
        return values


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if set(value) != {"data", "split", "model", "cost", "output"}:
        raise ValueError(
            "Config must contain exactly data, split, model, cost and output sections."
        )
    required = {
        "data": {"raw_dir"},
        "split": {"seed", "n_splits", "validation_fold"},
        "model": {"C", "max_iter", "tol"},
        "cost": {"false_positive", "false_negative"},
        "output": {"runs_dir"},
    }
    for section, keys in required.items():
        if set(value[section]) != keys:
            raise ValueError(f"Unexpected or missing keys in [{section}]; expected {sorted(keys)}.")
    root = path.resolve().parent.parent
    return Config(
        raw_dir=(root / value["data"]["raw_dir"]).resolve(),
        runs_dir=(root / value["output"]["runs_dir"]).resolve(),
        seed=value["split"]["seed"],
        n_splits=value["split"]["n_splits"],
        validation_fold=value["split"]["validation_fold"],
        C=value["model"]["C"],
        max_iter=value["model"]["max_iter"],
        tol=value["model"]["tol"],
        false_positive_cost=value["cost"]["false_positive"],
        false_negative_cost=value["cost"]["false_negative"],
    )

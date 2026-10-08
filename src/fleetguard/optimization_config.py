"""Bounded, sequential searches with explicit stability and learning-curve budgets."""

import math
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fleetguard.comparison_config import ComparisonConfig, load_comparison_config


@dataclass(frozen=True)
class OptimizationConfig:
    comparison: ComparisonConfig
    tracking_dir: Path
    xgb_trials: int = 6
    mlp_trials: int = 4
    sampler_seed: int = 44
    mlp_max_epochs: int = 24
    mlp_patience: int = 5
    mlp_inner_folds: int = 5
    control_max_iter: int = 100
    stability_seeds: tuple[int, ...] = (43, 44, 45)
    learning_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    device: str = "cpu"

    def __post_init__(self):
        for value in (
            self.xgb_trials,
            self.mlp_trials,
            self.mlp_max_epochs,
            self.mlp_patience,
            self.control_max_iter,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Trial and epoch budgets must be positive integers.")
        if type(self.mlp_inner_folds) is not int or self.mlp_inner_folds < 2:
            raise ValueError("MLP early-stopping split needs at least two folds.")
        seeds = (self.sampler_seed, *self.stability_seeds)
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Seeds must be unsigned 32-bit integers.")
        if not self.stability_seeds or len(set(self.stability_seeds)) != len(self.stability_seeds):
            raise ValueError("Stability seeds must be nonempty and unique.")
        fractions = self.learning_fractions
        if not fractions or len(set(fractions)) != len(fractions):
            raise ValueError("Learning fractions must be nonempty and unique.")
        if any(not math.isfinite(x) or not 0 < x <= 1 for x in fractions):
            raise ValueError("Learning fractions must lie in (0, 1].")
        if tuple(sorted(fractions)) != fractions or fractions[-1] != 1.0:
            raise ValueError("Learning fractions must increase and end at 1.0.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda; no implicit device switching.")

    def to_dict(self):
        value = asdict(self)
        value["comparison"] = self.comparison.to_dict()
        value["tracking_dir"] = str(self.tracking_dir)
        return value


def load_optimization_config(path: Path) -> OptimizationConfig:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    expected = {
        "comparison_config",
        "runs_dir",
        "tracking_dir",
        "xgb_trials",
        "mlp_trials",
        "sampler_seed",
        "mlp_max_epochs",
        "mlp_patience",
        "mlp_inner_folds",
        "control_max_iter",
        "stability_seeds",
        "learning_fractions",
        "device",
    }
    if set(value) != {"optimization"} or set(value["optimization"]) != expected:
        raise ValueError("Optimization config must contain exactly the documented keys.")
    values = dict(value["optimization"])
    root = path.resolve().parent.parent
    base_path = path.resolve().parent / values.pop("comparison_config")
    comparison = load_comparison_config(base_path)
    comparison = replace(
        comparison,
        base=replace(comparison.base, runs_dir=(root / values.pop("runs_dir")).resolve()),
    )
    values["tracking_dir"] = (root / values["tracking_dir"]).resolve()
    values["stability_seeds"] = tuple(values["stability_seeds"])
    values["learning_fractions"] = tuple(values["learning_fractions"])
    return OptimizationConfig(comparison=comparison, **values)

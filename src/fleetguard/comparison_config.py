"""Explicit budgets for the first train-only model comparison."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from fleetguard.config import Config

MODEL_NAMES = (
    "always_negative",
    "class_prior",
    "logistic",
    "logistic_balanced",
    "logistic_no_indicator",
    "logistic_log_robust",
    "random_forest",
    "hist_gradient_boosting",
    "hgb_sigmoid",
    "xgboost",
)


@dataclass(frozen=True)
class ComparisonConfig:
    base: Config
    cv_folds: int = 3
    seed: int = 43
    inner_folds: int = 5
    calibration_fold: int = 0
    threshold_fold: int = 1
    n_jobs: int = 2
    device: str = "cpu"
    models: tuple[str, ...] = MODEL_NAMES
    forest_estimators: int = 120
    forest_depth: int = 14
    forest_min_leaf: int = 3
    hgb_iterations: int = 150
    hgb_leaves: int = 15
    hgb_learning_rate: float = 0.1
    hgb_l2: float = 1.0
    xgb_estimators: int = 150
    xgb_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    xgb_l2: float = 1.0

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")
        if self.cv_folds < 2 or self.inner_folds < 3:
            raise ValueError("Comparison requires cv_folds >= 2 and inner_folds >= 3.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Comparison seed must be an unsigned 32-bit integer.")
        if (
            not 0 <= self.calibration_fold < self.inner_folds
            or not 0 <= self.threshold_fold < self.inner_folds
            or self.calibration_fold == self.threshold_fold
        ):
            raise ValueError("Calibration and threshold folds must be distinct valid folds.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Models must be a nonempty list without duplicates.")
        if set(self.models) - set(MODEL_NAMES):
            raise ValueError(f"Unknown models: {sorted(set(self.models) - set(MODEL_NAMES))}.")
        integers = (
            self.n_jobs,
            self.forest_estimators,
            self.forest_depth,
            self.forest_min_leaf,
            self.hgb_iterations,
            self.xgb_estimators,
            self.xgb_depth,
        )
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("Thread and tree budgets must be positive integers.")
        if self.hgb_leaves < 2:
            raise ValueError("hgb_leaves must be at least 2.")
        for value in (self.hgb_learning_rate, self.xgb_learning_rate):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rates must be positive and finite.")
        if any(
            not math.isfinite(x) or not 0 < x <= 1 for x in (self.xgb_subsample, self.xgb_colsample)
        ):
            raise ValueError("XGBoost sampling fractions must lie in (0, 1].")
        if any(not math.isfinite(x) or x < 0 for x in (self.hgb_l2, self.xgb_l2)):
            raise ValueError("L2 regularization must be finite and nonnegative.")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["base"] = self.base.to_dict()
        value["models"] = list(self.models)
        return value


def load_comparison_config(path: Path) -> ComparisonConfig:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    keys = {
        "data": {"raw_dir"},
        "output": {"runs_dir"},
        "split": {"seed", "n_splits", "validation_fold"},
        "comparison": {
            "cv_folds",
            "seed",
            "inner_folds",
            "calibration_fold",
            "threshold_fold",
            "n_jobs",
            "models",
        },
        "logistic": {"C", "max_iter", "tol"},
        "forest": {"n_estimators", "max_depth", "min_samples_leaf"},
        "boosting": {"max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization"},
        "xgboost": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
        },
        "cost": {"false_positive", "false_negative"},
    }
    if set(value) != set(keys):
        raise ValueError(f"Expected configuration sections: {sorted(keys)}.")
    for section, expected in keys.items():
        supplied = set(value[section])
        if section == "comparison":
            supplied -= {"device"}
        if supplied != expected:
            raise ValueError(
                f"Unexpected or missing keys in [{section}]; expected {sorted(expected)}."
            )
    root = path.resolve().parent.parent
    base = Config(
        raw_dir=(root / value["data"]["raw_dir"]).resolve(),
        runs_dir=(root / value["output"]["runs_dir"]).resolve(),
        seed=value["split"]["seed"],
        n_splits=value["split"]["n_splits"],
        validation_fold=value["split"]["validation_fold"],
        C=value["logistic"]["C"],
        max_iter=value["logistic"]["max_iter"],
        tol=value["logistic"]["tol"],
        false_positive_cost=value["cost"]["false_positive"],
        false_negative_cost=value["cost"]["false_negative"],
    )
    return ComparisonConfig(
        base=base,
        **{k: v for k, v in value["comparison"].items() if k != "models"},
        models=tuple(value["comparison"]["models"]),
        forest_estimators=value["forest"]["n_estimators"],
        forest_depth=value["forest"]["max_depth"],
        forest_min_leaf=value["forest"]["min_samples_leaf"],
        hgb_iterations=value["boosting"]["max_iter"],
        hgb_leaves=value["boosting"]["max_leaf_nodes"],
        hgb_learning_rate=value["boosting"]["learning_rate"],
        hgb_l2=value["boosting"]["l2_regularization"],
        xgb_estimators=value["xgboost"]["n_estimators"],
        xgb_depth=value["xgboost"]["max_depth"],
        xgb_learning_rate=value["xgboost"]["learning_rate"],
        xgb_subsample=value["xgboost"]["subsample"],
        xgb_colsample=value["xgboost"]["colsample_bytree"],
        xgb_l2=value["xgboost"]["reg_lambda"],
    )

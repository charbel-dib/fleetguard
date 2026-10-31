"""Release 08: use the whole official training file, then score the official test once.

Earlier releases split the 60,000 training rows into fit, calibration and threshold roles,
so the final model never saw most positives. This protocol keeps a single boundary:

1. grouped 5-fold cross-validation on the full training file gives out-of-fold scores;
2. the decision threshold minimizes 10 x FP + 500 x FN on those scores only;
3. the final model is fit on all 60,000 rows, frozen with its threshold, then the
   official test is scored exactly once.

Nothing measured on the official test can change the model or the threshold.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from fleetguard.artifacts import save_model
from fleetguard.data import TEST_NAME, TRAIN_NAME, load_aps
from fleetguard.io import read_json, sha256_file, write_json

COSTS = {"false_positive": 10, "false_negative": 500}

# Published official-test results, for context only (IDA 2016 Industrial Challenge).
REFERENCE_RESULTS = [
    {"name": "IDA 2016 winner (Costa & Nascimento)", "cost": 9920, "fp": 542, "fn": 9},
    {
        "name": "IDA 2016 second place (Gondek, Hafner & Sampson)",
        "cost": 10900,
        "fp": 490,
        "fn": 12,
    },
]


@dataclass(frozen=True)
class FullTrainConfig:
    folds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    positive_weight: float = 20.0
    max_iter: int = 400
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    bootstrap_resamples: int = 1000
    split_seed: int = 0
    costs: dict = field(default_factory=lambda: dict(COSTS))


class SeedEnsemble(ClassifierMixin, BaseEstimator):
    """Average of histogram gradient boosting models that differ only by random seed."""

    def __init__(
        self,
        seeds=(0, 1, 2),
        positive_weight=20.0,
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
    ):
        self.seeds = seeds
        self.positive_weight = positive_weight
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization

    def _member(self, seed):
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            class_weight={0: 1.0, 1: self.positive_weight},
            early_stopping=False,
            random_state=seed,
        )

    def fit(self, features, target):
        # Columns with no observed value carry no information and break histogram binning.
        self.used_columns_ = [c for c in features.columns if features[c].notna().any()]
        used = features[self.used_columns_]
        self.models_ = [self._member(seed).fit(used, target) for seed in self.seeds]
        self.classes_ = self.models_[0].classes_
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = features.shape[1]
        return self

    def predict_proba(self, features):
        used = features[self.used_columns_]
        return np.mean([model.predict_proba(used) for model in self.models_], axis=0)

    def predict(self, features):
        return self.classes_[(self.predict_proba(features)[:, 1] >= 0.5).astype(int)]


def ensemble_from(config: FullTrainConfig) -> SeedEnsemble:
    return SeedEnsemble(
        seeds=config.seeds,
        positive_weight=config.positive_weight,
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
    )


def feature_groups(features: pd.DataFrame) -> np.ndarray:
    """Identical sensor rows share a group, so duplicates never straddle two folds."""
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


def outcome(target, flagged, costs=COSTS) -> dict:
    target = np.asarray(target).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int((flagged & target).sum())
    fp = int((flagged & ~target).sum())
    fn = int((~flagged & target).sum())
    return {
        "rows": int(target.size),
        "positives": int(target.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cost": fp * costs["false_positive"] + fn * costs["false_negative"],
        "recall": tp / max(int(target.sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "flagged_rate": float(flagged.mean()),
    }


def cost_threshold(target, scores, costs=COSTS) -> tuple[float, int]:
    """Exact cost-minimizing threshold over all distinct scores (flag when score >= t).

    Ties prefer fewer missed failures, then fewer false alarms, then the higher threshold.
    """
    target = np.asarray(target).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_target = scores[order], target[order]
    # Flagging the k highest scores; only cut where the score changes.
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(~sorted_target)
    cut = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp, fp, thresholds = tp[cut], fp[cut], sorted_scores[cut]
    fn = int(target.sum()) - tp
    cost = fp * costs["false_positive"] + fn * costs["false_negative"]
    # Also consider flagging nothing.
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), thresholds]
    fp = np.r_[0, fp]
    fn = np.r_[int(target.sum()), fn]
    cost = np.r_[int(target.sum()) * costs["false_negative"], cost]
    best = np.lexsort((-thresholds, fp, fn, cost))[0]
    return float(thresholds[best]), int(cost[best])


def out_of_fold_scores(features, target, config: FullTrainConfig) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=config.folds, shuffle=True, random_state=config.split_seed
    )
    groups = feature_groups(features)
    scores = np.full(len(target), np.nan)
    for train_index, held_out in splitter.split(features, target, groups):
        model = ensemble_from(config).fit(features.iloc[train_index], target.iloc[train_index])
        scores[held_out] = model.predict_proba(features.iloc[held_out])[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("Some rows received no out-of-fold score.")
    return scores


def bootstrap_intervals(target, flagged, resamples, seed=0, costs=COSTS) -> dict:
    rng = np.random.default_rng(seed)
    target = np.asarray(target).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    rows = target.size
    values = {"cost": [], "recall": [], "precision": []}
    for _ in range(resamples):
        pick = rng.integers(0, rows, rows)
        result = outcome(target[pick], flagged[pick], costs)
        for key in values:
            values[key].append(result[key])
    return {
        key: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
        for key, v in values.items()
    }


def freeze(raw_dir: Path, output: Path, config: FullTrainConfig | None = None) -> dict:
    """Fit and freeze model + threshold using the training file only."""
    config = config or FullTrainConfig()
    output.mkdir(parents=True, exist_ok=False)
    features, target = load_aps(raw_dir / TRAIN_NAME)
    scores = out_of_fold_scores(features, target, config)
    threshold, oof_cost = cost_threshold(target, scores, config.costs)
    pd.DataFrame({"row_id": np.arange(len(target)), "target": target, "oof_score": scores}).to_csv(
        output / "oof_scores.csv", index=False
    )
    model = ensemble_from(config).fit(features, target)
    frozen_at = datetime.now(UTC).isoformat()
    save_model(
        output / "model",
        model,
        metadata={
            "name": "hgb_seed_ensemble",
            "feature_names": list(features.columns),
            "threshold": threshold,
            "costs": config.costs,
            "calibration": "none",
            "score_semantics": "class-weighted boosting ranking score, not a probability",
            "release_id": output.name,
        },
    )
    decision = {
        "protocol": "fulltrain_v1",
        "release_id": output.name,
        "threshold": threshold,
        "threshold_source": "grouped_out_of_fold_training_scores",
        "official_test_used_for_selection": False,
        "frozen_at_utc": frozen_at,
        "train_sha256": sha256_file(raw_dir / TRAIN_NAME),
        "model_sha256": sha256_file(output / "model" / "pipeline.joblib"),
        "config": asdict(config),
        "out_of_fold": outcome(target, scores >= threshold, config.costs),
    }
    write_json(output / "decision.json", decision)
    return decision


def score_official_test(raw_dir: Path, output: Path) -> dict:
    """Score the official test once with the frozen model and threshold."""
    result_path = output / "official_test.json"
    if result_path.exists():
        raise ValueError("The official test was already scored for this release.")
    decision = read_json(output / "decision.json")
    if sha256_file(output / "model" / "pipeline.joblib") != decision["model_sha256"]:
        raise ValueError("The frozen model changed after the decision was recorded.")
    from fleetguard.artifacts import load_model

    model, metadata = load_model(output / "model")
    features, target = load_aps(raw_dir / TEST_NAME)
    scores = model.predict_proba(features[metadata["feature_names"]])[:, 1]
    flagged = scores >= decision["threshold"]
    config = decision["config"]
    result = {
        "release_id": decision["release_id"],
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "test_sha256": sha256_file(raw_dir / TEST_NAME),
        "threshold": decision["threshold"],
        "official_test": outcome(target, flagged, config["costs"]),
        "bootstrap_95": bootstrap_intervals(
            target, flagged, config["bootstrap_resamples"], costs=config["costs"]
        ),
        "references": REFERENCE_RESULTS,
    }
    write_json(result_path, result)
    pd.DataFrame({"row_id": np.arange(len(target)), "target": target, "score": scores}).to_csv(
        output / "official_test_scores.csv", index=False
    )
    return result


def report(output: Path) -> str:
    decision = read_json(output / "decision.json")
    test = read_json(output / "official_test.json")
    t, oof, ci = test["official_test"], decision["out_of_fold"], test["bootstrap_95"]
    rows = [
        f"| **FleetGuard release 08 (this run)** | **{t['cost']:,}** | {t['fp']} | {t['fn']} |",
        *[f"| {r['name']} | {r['cost']:,} | {r['fp']} | {r['fn']} |" for r in test["references"]],
    ]
    lines = [
        f"# FleetGuard release 08 — full-train protocol (`{decision['release_id']}`)",
        "",
        "Model and threshold were frozen from the training file only "
        f"({decision['frozen_at_utc']}); the official test was scored once afterwards.",
        "",
        "## Official test (16,000 rows, 375 APS failures)",
        "",
        "| Model | Cost (10·FP + 500·FN) | False alarms | Missed failures |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        f"- Recall **{t['recall']:.2%}** "
        f"(95% bootstrap: {ci['recall'][0]:.2%}–{ci['recall'][1]:.2%})",
        f"- Precision {t['precision']:.2%} "
        f"(95% bootstrap: {ci['precision'][0]:.2%}–{ci['precision'][1]:.2%})",
        f"- Cost 95% bootstrap interval: {ci['cost'][0]:,.0f}–{ci['cost'][1]:,.0f}",
        f"- Flagged trucks: {t['flagged_rate']:.2%}",
        "",
        "A missed failure costs as much as 50 false alarms, so the cost-optimal policy",
        "flags widely: low precision is the intended trade-off, not a defect.",
        "",
        "## How the threshold was chosen",
        "",
        f"Grouped {decision['config']['folds']}-fold cross-validation on all 60,000 training rows "
        "(identical sensor rows stay in one fold). The threshold "
        f"`{decision['threshold']:.6g}` minimizes the challenge cost on out-of-fold scores: "
        f"cost {oof['cost']:,}, {oof['fn']} missed failures and {oof['fp']} false alarms "
        "on training.",
        "",
        "## Model",
        "",
        "Average of three histogram gradient boosting classifiers (scikit-learn), seeds "
        f"{list(decision['config']['seeds'])}, positive class weight "
        f"{decision['config']['positive_weight']:g}, {decision['config']['max_iter']} iterations, "
        f"learning rate {decision['config']['learning_rate']}. "
        "Hyperparameters were fixed a priori, not tuned on the official test.",
        "",
        "## Limits",
        "",
        "- One public, anonymized snapshot: results need validation before use on another fleet.",
        "- Scores are ranking scores, not calibrated probabilities.",
        "- Reference costs come from the published challenge results; protocols may differ.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    return text

"""Audit an existing XGBoost champion and freeze it without reading official test labels."""

import logging
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits

from fleetguard.artifacts import load_model, positive_scores, prepare_input, save_model
from fleetguard.comparison_splits import grouped_folds
from fleetguard.data import TRAIN_NAME, load_aps
from fleetguard.diagnostics import error_by_missingness, reliability_table
from fleetguard.download import verify_raw
from fleetguard.experiment import _environment, require_complete
from fleetguard.io import read_json, sha256_file, write_json
from fleetguard.metrics import classification_metrics, select_threshold, threshold_curve
from fleetguard.release_contract import freeze_release

logger = logging.getLogger("fleetguard")


def validated_roles(features, parent):
    """Use the recorded assignments, never re-split using validation labels."""
    split = pd.read_csv(parent / "split.csv", dtype={"feature_group": "uint64"})
    roles = pd.read_csv(parent / "final_roles.csv", dtype={"feature_group": "uint64"})
    if (
        len(split) != len(features)
        or split.row_id.duplicated().any()
        or set(split.row_id) != set(range(len(features)))
        or set(split.partition) != {"train", "validation"}
        or roles.row_id.duplicated().any()
        or set(roles.role) != {"fit", "calibration", "threshold"}
        or set(roles.row_id) != set(split.loc[split.partition == "train", "row_id"])
    ):
        raise ValueError("Release source role assignments are inconsistent.")
    groups = pd.util.hash_pandas_object(features, index=False)
    for table in (split, roles):
        if not np.array_equal(groups.iloc[table.row_id].to_numpy(), table.feature_group.to_numpy()):
            raise ValueError("Release feature groups differ from recorded data.")
    joined = split[["row_id", "partition", "feature_group"]].merge(
        roles[["row_id", "role"]], on="row_id", how="left"
    )
    joined["role"] = joined.role.fillna("validation")
    if joined.groupby("feature_group").role.nunique().max() != 1:
        raise ValueError("A feature group crosses release roles.")
    result = {name: roles.loc[roles.role == name, "row_id"].to_numpy() for name in set(roles.role)}
    result["validation"] = split.loc[split.partition == "validation", "row_id"].to_numpy()
    return result


def _sigmoid(model, x, y):
    result = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", n_jobs=1)
    result.fit(x, y)
    # A reversed calibration map would change the champion's ordering.
    if result.calibrated_classifiers_[0].calibrators[0].a_ >= 0:
        raise ValueError("Sigmoid calibration reversed the champion ranking.")
    return result


def choose_calibration(model, features, target, row_ids, config):
    raw_scores = positive_scores(model, features)
    heldout_scores = np.empty(len(features))
    audit = []
    for fold, (fit, score) in enumerate(
        grouped_folds(features, target, n_splits=config.calibration_folds, seed=config.seed)
    ):
        calibrated = _sigmoid(model, features.iloc[fit], target.iloc[fit])
        heldout_scores[score] = positive_scores(calibrated, features.iloc[score])
        for role, indices in (("calibrator_fit", fit), ("calibrator_score", score)):
            audit.append(pd.DataFrame({"cv_fold": fold, "role": role, "row_id": row_ids[indices]}))
    # This CV chooses ONLY the calibration map, never a new base model.
    rows = pd.DataFrame(
        [
            {
                "method": name,
                "brier_score": float(brier_score_loss(target, scores)),
                "average_precision": float(average_precision_score(target, scores)),
                "rows": len(target),
            }
            for name, scores in (("raw", raw_scores), ("sigmoid", heldout_scores))
        ]
    )
    method = "sigmoid" if rows.iloc[1].brier_score < rows.iloc[0].brier_score else "raw"
    selected = _sigmoid(model, features, target) if method == "sigmoid" else model
    np.testing.assert_array_equal(raw_scores, positive_scores(model, features))
    return selected, method, rows, pd.concat(audit, ignore_index=True)


def policy_thresholds(target, scores, costs, budgets):
    curve = threshold_curve(target, scores, **costs)
    rows = []
    policies = [("challenge_cost", None), *[(f"budget_{b:g}", b) for b in budgets]]
    for name, budget in policies:
        eligible = curve if budget is None else curve[(curve.tp + curve.fp) / len(target) <= budget]
        threshold = select_threshold(eligible)
        rows.append(
            {
                "policy": name,
                "budget_on_threshold_role": budget,
                **classification_metrics(target, scores, threshold=threshold, **costs),
            }
        )
    return pd.DataFrame(rows), curve


def _error_sensor_summary(features, prediction):
    rows = []
    for outcome in ("tp", "tn", "fp", "fn"):
        subset = features.loc[prediction.outcome.to_numpy() == outcome]
        for name in features.columns:
            rows.append(
                {
                    "outcome": outcome,
                    "feature": name,
                    "rows": len(subset),
                    "nonmissing_rows": int(subset[name].notna().sum()),
                    "median": subset[name].median() if subset[name].notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _importance(model, x, y, threshold, costs, *, repeats, seed):
    def negative_cost(estimator, features, target):
        p = positive_scores(estimator, features)
        yy = np.asarray(target)
        cost = costs["false_positive_cost"] * ((p >= threshold) & (yy == 0)).sum()
        cost += costs["false_negative_cost"] * ((p < threshold) & (yy == 1)).sum()
        return -float(cost / len(yy))

    values = permutation_importance(
        model,
        x,
        y,
        scoring={"cost": negative_cost, "ap": "average_precision"},
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    columns = pd.DataFrame(
        {
            "feature": x.columns,
            "cost_increase_per_row_mean": values["cost"].importances_mean,
            "cost_increase_per_row_std": values["cost"].importances_std,
            "average_precision_drop_mean": values["ap"].importances_mean,
            "average_precision_drop_std": values["ap"].importances_std,
            "repeats": repeats,
            "rows": len(x),
        }
    ).sort_values("cost_increase_per_row_mean", ascending=False)
    # Histogram-like names are anonymized; jointly shuffle multi-column prefixes.
    families = {}
    for name in x.columns:
        families.setdefault(name.rsplit("_", 1)[0], []).append(name)
    base_cost, base_ap = (
        negative_cost(model, x, y),
        average_precision_score(y, positive_scores(model, x)),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for prefix, names in sorted(families.items()):
        if len(names) < 2:
            continue
        cost_changes, ap_changes = [], []
        for _ in range(repeats):
            changed = x.copy()
            changed.loc[:, names] = x.iloc[rng.permutation(len(x))][names].to_numpy()
            cost_changes.append(base_cost - negative_cost(model, changed, y))
            ap_changes.append(base_ap - average_precision_score(y, positive_scores(model, changed)))
        rows.append(
            {
                "prefix": prefix,
                "features": ",".join(names),
                "columns": len(names),
                "cost_increase_per_row_mean": float(np.mean(cost_changes)),
                "cost_increase_per_row_std": float(np.std(cost_changes)),
                "average_precision_drop_mean": float(np.mean(ap_changes)),
                "repeats": repeats,
                "rows": len(x),
            }
        )
    return columns, pd.DataFrame(rows)


def run_audit(features, target, config, parent):
    source_run = require_complete(parent)
    champion = read_json(parent / "champion.json")
    if source_run.get("protocol") != "optimization_v1" or champion["model"] != "xgboost_tuned":
        raise ValueError("Audit v1 requires the uncalibrated xgboost_tuned optimization champion.")
    raw_model, original = load_model(parent / champion["model"])
    if original.get("calibration") is not None:
        raise ValueError("Audit requires a base model that has not used the calibration role.")
    if not features.index.equals(target.index) or not np.isin(target, [0, 1]).all():
        raise ValueError("Release input requires aligned binary labels.")
    features = prepare_input(features, original["feature_names"])
    roles = validated_roles(features, parent)
    if original["training_rows"] != len(roles["fit"]):
        raise ValueError("Base model fit count differs from the recorded role.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = config.releases_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "id": run_id,
        "kind": source_run["kind"],
        "protocol": "release_v1",
        "status": "running",
    }
    write_json(output / "run.json", status)
    try:
        for path in ("source.json", "split.csv", "final_roles.csv"):
            shutil.copyfile(parent / path, output / path)
        write_json(output / "config.json", config.to_dict())
        write_json(output / "environment.json", _environment())
        with threadpool_limits(limits=1):
            cal = roles["calibration"]
            selected, method, calibration, cal_roles = choose_calibration(
                raw_model, features.iloc[cal], target.iloc[cal], cal, config
            )
            calibration.to_csv(output / "calibration_cv.csv", index=False)
            cal_roles.to_csv(output / "calibration_roles.csv", index=False)
            tuning = roles["threshold"]
            policies, curve = policy_thresholds(
                target.iloc[tuning],
                positive_scores(selected, features.iloc[tuning]),
                original["costs"],
                config.inspection_budgets,
            )
            policies.to_csv(output / "policy_thresholds.csv", index=False)
            threshold = float(
                policies.loc[policies.policy == "challenge_cost", "threshold"].iloc[0]
            )
            name = "xgboost_release"
            # This record precedes all validation diagnostics and final-test access.
            decision = {
                "source_run": parent.name,
                "source_champion": champion["model"],
                "source_pipeline_sha256": original["pipeline_sha256"],
                "model": name,
                "calibration": method,
                "calibration_selection": "pooled grouped calibration-role OOF Brier; raw wins ties",
                "policy": "challenge_cost",
                "threshold": threshold,
                "threshold_selection": "threshold role only; min cost, FN, FP, highest threshold",
                "costs": original["costs"],
                "parameters": original["parameters"],
                "frozen_at_utc": datetime.now(UTC).isoformat(),
                "base_model_refitted": False,
                "validation_used_for_this_decision": False,
                "validation_previously_observed_in_development": True,
                "official_test_used_for_selection": False,
            }
            write_json(output / "decision.json", decision)
            write_json(
                output / "champion.json", {"model": name, "selection_partition": "development"}
            )
            metadata = {
                "model": name,
                "run_id": run_id,
                "kind": source_run["kind"],
                "protocol": "release_v1",
                "feature_names": original["feature_names"],
                "threshold": threshold,
                "costs": original["costs"],
                "parameters": original["parameters"],
                "xgboost_version": original["xgboost_version"],
                "training_rows": len(roles["fit"]),
                "calibration_rows": len(cal),
                "threshold_rows": len(tuning),
                "calibration": method,
                "threshold_source": "development_threshold_role",
                "score_semantics": (
                    "APS positive-class score; calibration assessed only on this snapshot"
                ),
                "release_id": run_id,
            }
            save_model(output / name, selected, metadata=metadata)
            curve.to_csv(output / name / "threshold_curve.csv", index=False)
            freeze_release(output, model=name)
            logger.info("Decision frozen before validation diagnostics: %s", output)
            val = roles["validation"]
            x_val, y_val = features.iloc[val], target.iloc[val]
            scores, raw_scores = positive_scores(selected, x_val), positive_scores(raw_model, x_val)
            metrics = classification_metrics(
                y_val, scores, threshold=threshold, **original["costs"]
            )
            raw_metrics = classification_metrics(
                y_val, raw_scores, threshold=original["threshold"], **original["costs"]
            )
            write_json(
                output / "validation_metrics.json", {"release": metrics, "source_raw": raw_metrics}
            )
            policy_rows = []
            for row in policies.itertuples(index=False):
                policy_rows.append(
                    {
                        "policy": row.policy,
                        "budget_on_threshold_role": row.budget_on_threshold_role,
                        **classification_metrics(
                            y_val, scores, threshold=row.threshold, **original["costs"]
                        ),
                    }
                )
            pd.DataFrame(policy_rows).to_csv(output / "validation_policies.csv", index=False)
            prediction = pd.DataFrame(
                {
                    "row_id": val,
                    "target": y_val.to_numpy(),
                    "score": scores,
                    "raw_score": raw_scores,
                    "prediction": (scores >= threshold).astype(int),
                    "missing_fraction": x_val.isna().mean(axis=1).to_numpy(),
                    "margin_from_threshold": scores - threshold,
                }
            )
            prediction["outcome"] = np.select(
                [
                    (prediction.target == 1) & (prediction.prediction == 1),
                    (prediction.target == 0) & (prediction.prediction == 0),
                    (prediction.target == 0) & (prediction.prediction == 1),
                ],
                ["tp", "tn", "fp"],
                default="fn",
            )
            prediction.to_csv(output / "validation_predictions.csv", index=False)
            prediction[prediction.outcome.isin(["fp", "fn"])].to_csv(
                output / "validation_errors.csv", index=False
            )
            _error_sensor_summary(x_val, prediction).to_csv(
                output / "error_sensor_summary.csv", index=False
            )
            error_by_missingness(
                y_val,
                scores,
                prediction.missing_fraction,
                threshold=threshold,
                costs=original["costs"],
            ).to_csv(output / "errors_by_missingness.csv", index=False)
            for label, values in (("raw", raw_scores), ("release", scores)):
                reliability_table(y_val, values).to_csv(
                    output / f"reliability_{label}.csv", index=False
                )
            logger.info(
                "Computing permutation audit: %s features x %s repeats",
                len(features.columns),
                config.permutation_repeats,
            )
            columns, groups = _importance(
                selected,
                x_val,
                y_val,
                threshold,
                original["costs"],
                repeats=config.permutation_repeats,
                seed=config.seed,
            )
            columns.to_csv(output / "permutation_columns.csv", index=False)
            groups.to_csv(output / "permutation_groups.csv", index=False)
        benchmark_input = output / "benchmark_input.csv"
        # Development fit sensors only; repeat rows for a tiny software fixture if necessary.
        sample = features.iloc[np.resize(roles["fit"], max(config.batch_sizes))]
        sample.to_csv(benchmark_input, index=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fleetguard.inference_benchmark",
                "--model",
                str(output / name),
                "--input",
                str(benchmark_input),
                "--output",
                str(output / "benchmark.json"),
                "--repeats",
                str(config.benchmark_repeats),
                "--batch-sizes",
                *map(str, config.batch_sizes),
            ],
            check=True,
        )
        benchmark_input.unlink()
        from fleetguard.release_report import write_release_report

        write_release_report(output)
        write_json(
            output / "run.json",
            {
                **status,
                "status": "complete",
                "split_sha256": sha256_file(output / "split.csv"),
                "freeze_sha256": sha256_file(output / "freeze.json"),
            },
        )
    except BaseException as exc:
        write_json(
            output / "run.json",
            {**status, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    return output


def audit(config, parent):
    manifest = verify_raw(config.raw_dir)
    if manifest["files"] != read_json(parent / "source.json")["files"]:
        raise ValueError("Data snapshot differs from the source optimization run.")
    features, target = load_aps(config.raw_dir / TRAIN_NAME)
    return run_audit(features, target, config, parent)

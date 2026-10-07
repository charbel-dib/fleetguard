"""Compare fixed candidates using train-only CV and separate calibration/tuning roles."""

import importlib.metadata
import logging
import time
import uuid
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from fleetguard.artifacts import positive_scores, save_model
from fleetguard.comparison_config import ComparisonConfig
from fleetguard.comparison_models import calibrate_frozen, make_candidates
from fleetguard.comparison_splits import grouped_folds, role_split
from fleetguard.data import TRAIN_NAME, load_aps, profile
from fleetguard.diagnostics import error_by_missingness, reliability_table
from fleetguard.download import verify_raw
from fleetguard.experiment import _environment
from fleetguard.io import sha256_file, write_json
from fleetguard.metrics import classification_metrics, select_threshold, threshold_curve
from fleetguard.splits import make_split

logger = logging.getLogger("fleetguard")


def _costs(config: ComparisonConfig) -> dict:
    return {
        "false_positive_cost": config.base.false_positive_cost,
        "false_negative_cost": config.base.false_negative_cost,
    }


def _roles(features, target, config, *, seed):
    return role_split(
        features,
        target,
        n_splits=config.inner_folds,
        seed=seed,
        calibration_fold=config.calibration_fold,
        threshold_fold=config.threshold_fold,
    )


def fit_and_tune(features, target, config: ComparisonConfig, roles: dict) -> dict:
    """No scoring partition is accepted here; its labels cannot affect fit or thresholds."""
    x_fit, y_fit = features.iloc[roles["fit"]], target.iloc[roles["fit"]]
    x_cal, y_cal = features.iloc[roles["calibration"]], target.iloc[roles["calibration"]]
    x_tune, y_tune = features.iloc[roles["threshold"]], target.iloc[roles["threshold"]]
    fitted, durations = {}, {}
    with threadpool_limits(limits=config.n_jobs):
        for name, model in make_candidates(config).items():
            logger.info("Fitting %s on %s observations", name, len(x_fit))
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x_fit, y_fit)
            except ConvergenceWarning as exc:
                raise RuntimeError(
                    f"{name} did not converge; no candidate will be published."
                ) from exc
            durations[name] = time.perf_counter() - started
            fitted[name] = model
        if "hgb_sigmoid" in config.models:
            started = time.perf_counter()
            fitted["hgb_sigmoid"] = calibrate_frozen(fitted["hist_gradient_boosting"], x_cal, y_cal)
            durations["hgb_sigmoid"] = (
                durations["hist_gradient_boosting"] + time.perf_counter() - started
            )
        result = {}
        for name in config.models:
            score = positive_scores(fitted[name], x_tune)
            curve = threshold_curve(y_tune, score, **_costs(config))
            threshold = 0.5 if name == "always_negative" else select_threshold(curve)
            result[name] = {
                "model": fitted[name],
                "threshold": threshold,
                "curve": curve,
                "tuning_metrics": classification_metrics(
                    y_tune, score, threshold=threshold, **_costs(config)
                ),
                "fit_seconds": durations[name],
            }
    return result


def summarize_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only scoring-fold metrics; never tuning-set costs."""
    selected = fold_metrics[fold_metrics["rule"] == "frozen_threshold"]
    rows = []
    for name, group in selected.groupby("model", sort=True):
        default = fold_metrics[
            (fold_metrics["model"] == name) & (fold_metrics["rule"] == "threshold_0.5")
        ]
        rows.append(
            {
                "model": name,
                "folds": len(group),
                "scored_rows": int(group["rows"].sum()),
                "cost": int(group["cost"].sum()),
                "cost_per_row": float(group["cost"].sum() / group["rows"].sum()),
                "mean_fold_cost_per_row": float(group["cost_per_row"].mean()),
                "std_fold_cost_per_row": float(group["cost_per_row"].std(ddof=1)),
                "mean_average_precision": float(group["average_precision"].mean()),
                "std_average_precision": float(group["average_precision"].std(ddof=1)),
                "mean_recall": float(group["recall"].mean()),
                "mean_precision": float(group["precision"].mean()),
                "mean_brier_score": float(group["brier_score"].mean()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "fp": int(group["fp"].sum()),
                "fn": int(group["fn"].sum()),
                "tp": int(group["tp"].sum()),
                "tn": int(group["tn"].sum()),
                "cost_at_0_5": int(default["cost"].sum()),
                "mean_fit_seconds": float(group["fit_seconds"].mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["cost_per_row", "mean_average_precision", "model"], ascending=[True, False, True]
        )
        .reset_index(drop=True)
    )


def run_comparison(
    features, target, config: ComparisonConfig, *, source: dict, kind: str = "official_aps"
) -> Path:
    development, validation, assignments = make_split(
        features,
        target,
        seed=config.base.seed,
        n_splits=config.base.n_splits,
        validation_fold=config.base.validation_fold,
    )
    x_development, y_development = features.iloc[development], target.iloc[development]
    cv_folds = grouped_folds(
        x_development, y_development, n_splits=config.cv_folds, seed=config.seed
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = config.base.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    assignments.to_csv(run_dir / "split.csv", index=False)
    write_json(run_dir / "source.json", source)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "data_profile.json", profile(features, target))
    environment = _environment()
    for package in ("matplotlib", "xgboost-cpu", "xgboost"):
        try:
            environment["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "run.json",
        {"id": run_id, "kind": kind, "protocol": "comparison_v1", "status": "running"},
    )
    metric_rows, oof_tables, role_tables = [], [], []
    try:
        for fold, (pool, scoring) in enumerate(cv_folds):
            logger.info("CV fold %s/%s", fold + 1, config.cv_folds)
            x_pool, y_pool = x_development.iloc[pool], y_development.iloc[pool]
            roles, role_assignments = _roles(x_pool, y_pool, config, seed=config.seed + fold + 1)
            role_assignments["row_id"] = development[pool]
            role_assignments["cv_fold"] = fold
            scored_roles = pd.DataFrame(
                {
                    "row_id": development[scoring],
                    "cv_fold": fold,
                    "role": "score",
                    "feature_group": pd.util.hash_pandas_object(
                        x_development.iloc[scoring], index=False
                    ).to_numpy(),
                }
            )
            role_tables.extend([role_assignments.drop(columns="local_row"), scored_roles])
            fitted = fit_and_tune(x_pool, y_pool, config, roles)
            x_score, y_score = x_development.iloc[scoring], y_development.iloc[scoring]
            with threadpool_limits(limits=config.n_jobs):
                for name, bundle in fitted.items():
                    scores = positive_scores(bundle["model"], x_score)
                    for rule, threshold in (
                        ("frozen_threshold", bundle["threshold"]),
                        ("threshold_0.5", 0.5),
                    ):
                        metric_rows.append(
                            {
                                "model": name,
                                "fold": fold,
                                "rule": rule,
                                "fit_seconds": bundle["fit_seconds"],
                                **classification_metrics(
                                    y_score, scores, threshold=threshold, **_costs(config)
                                ),
                            }
                        )
                    oof_tables.append(
                        pd.DataFrame(
                            {
                                "model": name,
                                "fold": fold,
                                "row_id": development[scoring],
                                "target": y_score.to_numpy(),
                                "score": scores,
                                "threshold": bundle["threshold"],
                                "prediction": (scores >= bundle["threshold"]).astype(int),
                                "missing_fraction": x_score.isna().mean(axis=1).to_numpy(),
                            }
                        )
                    )
            logger.info("Completed scoring fold %s", fold + 1)
        folds = pd.DataFrame(metric_rows)
        summary = summarize_cv(folds)
        folds.to_csv(run_dir / "cv_fold_metrics.csv", index=False)
        summary.to_csv(run_dir / "cv_summary.csv", index=False)
        pd.concat(oof_tables, ignore_index=True).to_csv(
            run_dir / "oof_predictions.csv", index=False
        )
        pd.concat(role_tables, ignore_index=True).to_csv(run_dir / "cv_roles.csv", index=False)
        # Selection occurs before any prediction or evaluation on the retained validation set.
        champion = str(summary.iloc[0]["model"])
        write_json(
            run_dir / "champion.json",
            {
                "model": champion,
                "selection_partition": "development_cross_validation",
                "selection_metric": "scoring_cost_per_row_then_mean_fold_average_precision",
            },
        )
        final_roles, final_assignments = _roles(
            x_development, y_development, config, seed=config.seed
        )
        final_assignments["row_id"] = development
        final_assignments.drop(columns="local_row").to_csv(run_dir / "final_roles.csv", index=False)
        logger.info("CV-selected champion: %s; fitting its frozen deployment candidate", champion)
        bundle = fit_and_tune(
            x_development, y_development, replace(config, models=(champion,)), final_roles
        )[champion]
        with threadpool_limits(limits=config.n_jobs):
            scores = positive_scores(bundle["model"], features.iloc[validation])
        metrics = classification_metrics(
            target.iloc[validation], scores, threshold=bundle["threshold"], **_costs(config)
        )
        default = classification_metrics(
            target.iloc[validation], scores, threshold=0.5, **_costs(config)
        )
        metadata = {
            "model": champion,
            "run_id": run_id,
            "kind": kind,
            "protocol": "comparison_v1",
            "feature_names": features.columns.tolist(),
            "threshold": bundle["threshold"],
            "costs": _costs(config),
            "validation": metrics,
            "validation_at_0_5": default,
            "threshold_tuning": bundle["tuning_metrics"],
            "training_rows": len(final_roles["fit"]),
            "calibration_rows": len(final_roles["calibration"]),
            "calibration_role_used": champion == "hgb_sigmoid",
            "threshold_rows": len(final_roles["threshold"]),
            "validation_rows": len(validation),
            "fit_seconds": bundle["fit_seconds"],
            "calibration": "sigmoid_on_separate_calibration_role"
            if champion == "hgb_sigmoid"
            else None,
            "score_semantics": "positive-class score; population calibration is not established",
        }
        if champion == "xgboost":
            import xgboost

            metadata["xgboost_version"] = xgboost.__version__
        save_model(run_dir / champion, bundle["model"], metadata=metadata)
        bundle["curve"].to_csv(run_dir / champion / "threshold_curve.csv", index=False)
        prediction = pd.DataFrame(
            {
                "row_id": validation,
                "target": target.iloc[validation].to_numpy(),
                "score": scores,
                "prediction": (scores >= bundle["threshold"]).astype(int),
                "missing_fraction": features.iloc[validation].isna().mean(axis=1).to_numpy(),
            }
        )
        prediction.to_csv(run_dir / champion / "validation_predictions.csv", index=False)
        reliability_table(prediction["target"], prediction["score"]).to_csv(
            run_dir / "reliability.csv", index=False
        )
        errors = error_by_missingness(
            prediction["target"],
            prediction["score"],
            prediction["missing_fraction"],
            threshold=bundle["threshold"],
            costs=_costs(config),
        )
        errors.to_csv(run_dir / "errors_by_missingness.csv", index=False)
        prediction[prediction["target"] != prediction["prediction"]].to_csv(
            run_dir / "error_cases.csv", index=False
        )
        write_json(
            run_dir / "validation_metrics.json",
            {"model": champion, "frozen_threshold": metrics, "threshold_0.5": default},
        )
        from fleetguard.comparison_report import write_comparison_report

        write_comparison_report(run_dir, summary, prediction, metadata, errors)
        write_json(
            run_dir / "run.json",
            {
                "id": run_id,
                "kind": kind,
                "protocol": "comparison_v1",
                "status": "complete",
                "split_sha256": sha256_file(run_dir / "split.csv"),
                "cv_roles_sha256": sha256_file(run_dir / "cv_roles.csv"),
                "final_roles_sha256": sha256_file(run_dir / "final_roles.csv"),
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "run.json",
            {
                "id": run_id,
                "kind": kind,
                "protocol": "comparison_v1",
                "status": "failed",
                "error": str(exc),
            },
        )
        raise
    return run_dir


def compare(config: ComparisonConfig) -> Path:
    source = verify_raw(config.base.raw_dir)
    features, target = load_aps(config.base.raw_dir / TRAIN_NAME)
    return run_comparison(features, target, config, source=source)

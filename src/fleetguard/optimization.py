"""Budgeted train-only search. The official test is never read or selected against."""

import importlib.metadata
import logging
import time
import uuid
import warnings
from datetime import UTC, datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from fleetguard.artifacts import positive_scores, save_model
from fleetguard.comparison import _costs, summarize_cv
from fleetguard.comparison_models import calibrate_frozen
from fleetguard.comparison_splits import grouped_folds, role_split
from fleetguard.data import TRAIN_NAME, load_aps, profile
from fleetguard.diagnostics import error_by_missingness, reliability_table
from fleetguard.download import verify_raw
from fleetguard.experiment import _environment
from fleetguard.io import sha256_file, write_json
from fleetguard.metrics import classification_metrics, select_threshold, threshold_curve
from fleetguard.optimization_config import OptimizationConfig
from fleetguard.optimization_models import ABLATIONS, baseline_xgb, make_candidate, suggest
from fleetguard.splits import make_split
from fleetguard.tracking import LocalTracker

logger = logging.getLogger("fleetguard")


def _roles(x, y, config, seed):
    c = config.comparison
    return role_split(
        x,
        y,
        n_splits=c.inner_folds,
        seed=seed,
        calibration_fold=c.calibration_fold,
        threshold_fold=c.threshold_fold,
    )


def subset_fit(features, target, fraction, *, seed):
    """Nested seeded subsets of whole feature groups, stratified by group positive label."""
    if fraction == 1:
        return np.arange(len(features))
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    table = (
        pd.DataFrame({"group": groups, "target": target.to_numpy()})
        .groupby("group")["target"]
        .max()
    )
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        candidates = table[table == label].index.to_numpy()
        order = rng.permutation(candidates)
        chosen.extend(order[: max(1, int(np.ceil(len(order) * fraction)))])
    return np.flatnonzero(np.isin(groups, chosen))


def fit_candidate(x, y, roles, family, params, config, *, fraction=1.0):
    """Fit/calibrate/tune only; no scoring labels or retained validation are accepted."""
    fit = roles["fit"]
    subset = subset_fit(x.iloc[fit], y.iloc[fit], fraction, seed=config.comparison.base.seed)
    fit = fit[subset]
    model = make_candidate(
        family, params, config, seed=config.comparison.base.seed, fit_target=y.iloc[fit]
    )
    start = time.perf_counter()
    with threadpool_limits(limits=config.comparison.n_jobs), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x.iloc[fit], y.iloc[fit])
        if family == "xgboost":
            # Threshold/scoring inputs are pandas frames; persist CPU inference settings.
            model.set_params(classifier__device="cpu")
        raw_model = model
        if family == "mlp":
            model = calibrate_frozen(
                model, x.iloc[roles["calibration"]], y.iloc[roles["calibration"]]
            )
        tuning_score = positive_scores(model, x.iloc[roles["threshold"]])
    curve = threshold_curve(y.iloc[roles["threshold"]], tuning_score, **_costs(config.comparison))
    threshold = select_threshold(curve)
    return dict(
        model=model,
        raw_model=raw_model,
        threshold=threshold,
        curve=curve,
        training_rows=len(fit),
        fit_rows=fit,
        fit_seconds=time.perf_counter() - start,
        tuning_metrics=classification_metrics(
            y.iloc[roles["threshold"]],
            tuning_score,
            threshold=threshold,
            **_costs(config.comparison),
        ),
    )


def evaluate_cv(x, y, config, family, params, name, *, seed=None, fraction=1.0):
    c = config.comparison
    seed = c.seed if seed is None else seed
    metrics, predictions, assignments, histories, epoch_roles = [], [], [], [], []
    for fold, (pool, score) in enumerate(grouped_folds(x, y, n_splits=c.cv_folds, seed=seed)):
        logger.info(
            "%s: CV fold %s/%s, training fraction %.2f", name, fold + 1, c.cv_folds, fraction
        )
        xp, yp = x.iloc[pool], y.iloc[pool]
        roles, role_assignments = _roles(xp, yp, config, seed + fold + 1)
        role_assignments["row_id"] = x.index.to_numpy()[pool]
        role_assignments["cv_fold"] = fold
        scored = pd.DataFrame(
            {
                "row_id": x.index.to_numpy()[score],
                "cv_fold": fold,
                "role": "score",
                "feature_group": pd.util.hash_pandas_object(x.iloc[score], index=False).to_numpy(),
            }
        )
        assignments.extend([role_assignments.drop(columns="local_row"), scored])
        bundle = fit_candidate(xp, yp, roles, family, params, config, fraction=fraction)
        with threadpool_limits(limits=c.n_jobs):
            scores = positive_scores(bundle["model"], x.iloc[score])
        for rule, threshold in (("frozen_threshold", bundle["threshold"]), ("threshold_0.5", 0.5)):
            metrics.append(
                dict(
                    model=name,
                    fold=fold,
                    rule=rule,
                    fit_seconds=bundle["fit_seconds"],
                    training_rows=bundle["training_rows"],
                    requested_fraction=fraction,
                    **classification_metrics(
                        y.iloc[score], scores, threshold=threshold, **_costs(c)
                    ),
                )
            )
        predictions.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": fold,
                    "row_id": x.index.to_numpy()[score],
                    "target": y.iloc[score].to_numpy(),
                    "score": scores,
                    "threshold": bundle["threshold"],
                    "prediction": (scores >= bundle["threshold"]).astype(int),
                }
            )
        )
        if family == "mlp":
            raw = bundle["raw_model"]
            for stage, history in (
                ("epoch_selection", raw.history_),
                ("refit", raw.refit_history_),
            ):
                histories.extend(
                    [
                        dict(
                            model=name,
                            fold=fold,
                            stage=stage,
                            selected_epochs=raw.selected_epochs_,
                            **row,
                        )
                        for row in history
                    ]
                )
            audit = raw.epoch_roles_.copy()
            audit["row_id"] = xp.index.to_numpy()[bundle["fit_rows"]][audit["local_row"]]
            audit["cv_fold"] = fold
            epoch_roles.append(audit.drop(columns="local_row"))
    folds = pd.DataFrame(metrics)
    return dict(
        summary=summarize_cv(folds).iloc[0].to_dict(),
        folds=folds,
        predictions=pd.concat(predictions, ignore_index=True),
        roles=pd.concat(assignments, ignore_index=True),
        history=pd.DataFrame(histories),
        epoch_roles=pd.concat(epoch_roles, ignore_index=True) if epoch_roles else pd.DataFrame(),
    )


def _write_candidate(directory, result, *, params):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "parameters.json", params)
    write_json(directory / "summary.json", result["summary"])
    result["folds"].to_csv(directory / "fold_metrics.csv", index=False)
    result["predictions"].to_csv(directory / "oof_predictions.csv", index=False)
    result["roles"].to_csv(directory / "cv_roles.csv", index=False)
    if not result["history"].empty:
        result["history"].to_csv(directory / "epoch_history.csv", index=False)
        result["epoch_roles"].to_csv(directory / "epoch_roles.csv", index=False)


def _search(x, y, config, family, directory, tracker, parent):
    budget = config.xgb_trials if family == "xgboost" else config.mlp_trials
    directory.mkdir()
    study = optuna.create_study(
        study_name=family,
        direction="minimize",
        storage="sqlite:///" + (directory / "optuna.db").resolve().as_posix(),
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed, n_startup_trials=2),
        pruner=optuna.pruners.NopPruner(),
    )
    if family == "xgboost":
        study.enqueue_trial(baseline_xgb(config))
    else:
        study.enqueue_trial(
            dict(
                width=64,
                layers=2,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.0001,
                batch_size=1024,
            )
        )
    results, parameters, trial_rows = {}, {}, []

    def objective(trial):
        params = suggest(trial, family)
        name = f"{family}_trial_{trial.number:03d}"
        tracking_run = tracker.start(
            name, parent=parent, tags={"family": family, "phase": "search"}
        )
        tracker.log(
            tracking_run,
            params={**params, "trial": trial.number, "cv_seed": config.comparison.seed},
        )
        try:
            result = evaluate_cv(x, y, config, family, params, name)
            _write_candidate(directory / name, result, params=params)
            tracker.log(tracking_run, metrics=result["summary"])
            tracker.artifact(tracking_run, directory / name / "fold_metrics.csv")
            tracker.finish(tracking_run)
        except BaseException:
            tracker.finish(tracking_run, failed=True)
            raise
        results[trial.number], parameters[trial.number] = result, params
        trial_rows.append(
            dict(
                family=family,
                trial=trial.number,
                mlflow_run_id=tracking_run,
                **params,
                **result["summary"],
            )
        )
        return result["summary"]["cost_per_row"]

    study.optimize(objective, n_trials=budget, n_jobs=1)
    # Explicit tie-break uses AP then trial number; never a validation result.
    selected = min(
        results,
        key=lambda n: (
            results[n]["summary"]["cost_per_row"],
            -results[n]["summary"]["mean_average_precision"],
            n,
        ),
    )
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(directory / "trials.csv", index=False)
    return parameters[selected], results[selected], trials, selected


def _diagnostics(x, y, config, specs, selected, run_dir):
    stability, learning, audit = [], [], []
    for name, spec in specs.items():
        if spec["family"] not in {"xgboost", "mlp"}:
            continue
        for seed in config.stability_seeds:
            result = (
                selected[name]
                if seed == config.comparison.seed
                else evaluate_cv(x, y, config, spec["family"], spec["params"], name, seed=seed)
            )
            stability.append(dict(seed=seed, **result["summary"]))
            audit.append(
                result["roles"].assign(model=name, diagnostic="stability", seed=seed, fraction=1.0)
            )
        for fraction in config.learning_fractions:
            result = (
                selected[name]
                if fraction == 1
                else evaluate_cv(
                    x, y, config, spec["family"], spec["params"], name, fraction=fraction
                )
            )
            for row in result["folds"].query("rule == 'frozen_threshold'").to_dict("records"):
                learning.append(dict(fraction=fraction, **row))
            audit.append(
                result["roles"].assign(
                    model=name,
                    diagnostic="learning",
                    seed=config.comparison.seed,
                    fraction=fraction,
                )
            )
    pd.DataFrame(stability).to_csv(run_dir / "stability.csv", index=False)
    pd.DataFrame(learning).to_csv(run_dir / "learning_curves.csv", index=False)
    pd.concat(audit, ignore_index=True).to_csv(run_dir / "diagnostic_roles.csv", index=False)


def run_optimization(features, target, config: OptimizationConfig, *, source, kind="official_aps"):
    c = config.comparison
    development, validation, split = make_split(
        features,
        target,
        seed=c.base.seed,
        n_splits=c.base.n_splits,
        validation_fold=c.base.validation_fold,
    )
    x, y = features.iloc[development], target.iloc[development]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = c.base.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    split.to_csv(run_dir / "split.csv", index=False)
    write_json(run_dir / "source.json", source)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "data_profile.json", profile(features, target))
    environment = _environment()
    for name in ("optuna", "mlflow", "torch", "imbalanced-learn", "matplotlib", "xgboost"):
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(run_dir / "environment.json", environment)
    status = dict(id=run_id, kind=kind, protocol="optimization_v1", status="running")
    write_json(run_dir / "run.json", status)
    tracker, parent, active_child = None, None, None
    try:
        tracker = LocalTracker(config.tracking_dir)
        parent = tracker.start(run_id, tags={"kind": kind, "phase": "parent"})
        write_json(
            run_dir / "tracking.json",
            dict(uri=tracker.uri, parent_run_id=parent, experiment_id=tracker.experiment_id),
        )
        tracker.log(
            parent,
            params=dict(
                xgb_trials=config.xgb_trials,
                mlp_trials=config.mlp_trials,
                cv_folds=c.cv_folds,
                device=config.device,
                sampler_seed=config.sampler_seed,
            ),
            tags={"official_test_evaluated": "false"},
        )
        specs, selected, all_trials = {}, {}, []
        for family in ("xgboost", "mlp"):
            params, result, trials, best_trial = _search(
                x, y, config, family, run_dir / family, tracker, parent
            )
            name = f"{family}_tuned"
            spec = dict(family=family, params=params, selected_trial=best_trial)
            specs[name] = spec
            result["summary"]["model"] = name
            for table in ("folds", "predictions", "history"):
                if not result[table].empty:
                    result[table]["model"] = name
            selected[name] = result
            _write_candidate(run_dir / "candidates" / name, result, params=params)
            all_trials.append(trials)
        # Fixed controls, including the original XGBoost and one-factor logistic ablations.
        controls = {"xgboost_reference": dict(family="xgboost", params=baseline_xgb(config))}
        controls.update({name: dict(family=name, params={}) for name in ABLATIONS})
        control_status = []
        for name, spec in controls.items():
            specs[name] = spec
            control_run = tracker.start(name, parent=parent, tags={"phase": "control"})
            active_child = control_run
            if name == "xgboost_reference":
                # Trial zero was enqueued with these exact parameters; reuse its actual results.
                folds = pd.read_csv(run_dir / "xgboost/xgboost_trial_000/fold_metrics.csv")
                folds["model"] = name
                result = dict(
                    summary=summarize_cv(folds).iloc[0].to_dict(),
                    folds=folds,
                    roles=pd.read_csv(run_dir / "xgboost/xgboost_trial_000/cv_roles.csv"),
                    predictions=pd.read_csv(
                        run_dir / "xgboost/xgboost_trial_000/oof_predictions.csv"
                    ),
                    history=pd.DataFrame(),
                    epoch_roles=pd.DataFrame(),
                )
            else:
                try:
                    result = evaluate_cv(x, y, config, spec["family"], spec["params"], name)
                except ConvergenceWarning as exc:
                    failure = dict(
                        model=name,
                        status="failed_convergence",
                        error=str(exc),
                        max_iter=config.control_max_iter,
                    )
                    control_status.append(failure)
                    write_json(run_dir / "candidates" / name / "failure.json", failure)
                    tracker.log(control_run, tags={"failure": str(exc)})
                    tracker.finish(control_run, failed=True)
                    active_child = None
                    logger.warning(
                        "Control %s excluded: did not converge within %s iterations.",
                        name,
                        config.control_max_iter,
                    )
                    continue
            selected[name] = result
            _write_candidate(run_dir / "candidates" / name, result, params=spec["params"])
            tracker.log(control_run, metrics=result["summary"])
            tracker.finish(control_run)
            active_child = None
            control_status.append(
                dict(
                    model=name,
                    status="complete",
                    error="",
                    max_iter=config.control_max_iter if name in ABLATIONS else None,
                )
            )
        pd.DataFrame(control_status).to_csv(run_dir / "control_status.csv", index=False)
        summary = (
            pd.DataFrame([result["summary"] for result in selected.values()])
            .sort_values(
                ["cost_per_row", "mean_average_precision", "model"], ascending=[True, False, True]
            )
            .reset_index(drop=True)
        )
        summary.to_csv(run_dir / "cv_summary.csv", index=False)
        pd.concat(all_trials, ignore_index=True).to_csv(run_dir / "trials.csv", index=False)
        first = next(iter(selected.values()))
        first["roles"].to_csv(run_dir / "cv_roles.csv", index=False)
        # Freeze the candidate before running descriptive diagnostics or accessing validation.
        champion = str(summary.iloc[0]["model"])
        write_json(
            run_dir / "champion.json",
            dict(
                model=champion,
                selection_partition="development_search_cross_validation",
                selection_metric="cost_per_row_then_mean_fold_average_precision",
                **specs[champion],
            ),
        )
        write_json(run_dir / "selected_parameters.json", specs)
        _diagnostics(
            x, y, config, {k: specs[k] for k in ("xgboost_tuned", "mlp_tuned")}, selected, run_dir
        )
        roles, assignments = _roles(x, y, config, c.seed)
        assignments["row_id"] = development
        assignments.drop(columns="local_row").to_csv(run_dir / "final_roles.csv", index=False)
        spec = specs[champion]
        bundle = fit_candidate(x, y, roles, spec["family"], spec["params"], config)
        with threadpool_limits(limits=c.n_jobs):
            scores = positive_scores(bundle["model"], features.iloc[validation])
        metrics = classification_metrics(
            target.iloc[validation], scores, threshold=bundle["threshold"], **_costs(c)
        )
        default = classification_metrics(
            target.iloc[validation], scores, threshold=0.5, **_costs(c)
        )
        metadata = dict(
            model=champion,
            run_id=run_id,
            kind=kind,
            protocol="optimization_v1",
            feature_names=features.columns.tolist(),
            threshold=bundle["threshold"],
            costs=_costs(c),
            validation=metrics,
            validation_at_0_5=default,
            threshold_tuning=bundle["tuning_metrics"],
            training_rows=bundle["training_rows"],
            calibration_rows=len(roles["calibration"]),
            threshold_rows=len(roles["threshold"]),
            validation_rows=len(validation),
            fit_seconds=bundle["fit_seconds"],
            parameters=spec["params"],
            calibration="sigmoid_on_separate_calibration_role" if spec["family"] == "mlp" else None,
            score_semantics="positive-class score; population calibration is not established",
        )
        if spec["family"] == "xgboost":
            import xgboost

            metadata["xgboost_version"] = xgboost.__version__
        if spec["family"] == "mlp":
            import torch

            metadata["torch_version"] = torch.__version__
            metadata["selected_epochs"] = bundle["raw_model"].selected_epochs_
            bundle["raw_model"].epoch_roles_.to_csv(run_dir / "final_epoch_roles.csv", index=False)
        save_model(run_dir / champion, bundle["model"], metadata=metadata)
        bundle["curve"].to_csv(run_dir / champion / "threshold_curve.csv", index=False)
        prediction = pd.DataFrame(
            dict(
                row_id=validation,
                target=target.iloc[validation].to_numpy(),
                score=scores,
                prediction=(scores >= bundle["threshold"]).astype(int),
                missing_fraction=features.iloc[validation].isna().mean(axis=1).to_numpy(),
            )
        )
        prediction.to_csv(run_dir / champion / "validation_predictions.csv", index=False)
        errors = error_by_missingness(
            prediction["target"],
            prediction["score"],
            prediction["missing_fraction"],
            threshold=bundle["threshold"],
            costs=_costs(c),
        )
        errors.to_csv(run_dir / "errors_by_missingness.csv", index=False)
        reliability_table(prediction["target"], prediction["score"]).to_csv(
            run_dir / "reliability.csv", index=False
        )
        write_json(
            run_dir / "validation_metrics.json",
            dict(model=champion, frozen_threshold=metrics, threshold_0_5=default),
        )
        from fleetguard.optimization_report import write_optimization_report

        write_optimization_report(run_dir, summary, prediction, metadata, errors)
        tracker.log(
            parent,
            metrics={"validation_cost": metrics["cost"], "validation_recall": metrics["recall"]},
            tags={"champion": champion},
        )
        for name in (
            "cv_summary.csv",
            "trials.csv",
            "stability.csv",
            "learning_curves.csv",
            "selected_parameters.json",
            "source.json",
            "environment.json",
            "config.json",
            "optimization_report.md",
            "control_status.csv",
        ):
            tracker.artifact(parent, run_dir / name)
        for path in (run_dir / "figures").glob("*.png"):
            tracker.artifact(parent, path)
        tracker.finish(parent)
        write_json(
            run_dir / "run.json",
            {
                **status,
                "status": "complete",
                "split_sha256": sha256_file(run_dir / "split.csv"),
                "cv_roles_sha256": sha256_file(run_dir / "cv_roles.csv"),
                "final_roles_sha256": sha256_file(run_dir / "final_roles.csv"),
            },
        )
    except BaseException as exc:
        if tracker is not None and parent is not None:
            if active_child is not None:
                tracker.finish(active_child, failed=True)
            tracker.finish(parent, failed=True)
        write_json(
            run_dir / "run.json",
            {**status, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    return run_dir


def optimize(config):
    source = verify_raw(config.comparison.base.raw_dir)
    features, target = load_aps(config.comparison.base.raw_dir / TRAIN_NAME)
    return run_optimization(features, target, config, source=source)

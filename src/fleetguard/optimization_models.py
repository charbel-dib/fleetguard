"""Two search spaces and controlled preprocessing/resampling ablations."""

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ResamplingPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from fleetguard.comparison_models import signed_log1p
from fleetguard.torch_model import TorchMLPClassifier

ABLATIONS = ("logistic", "logistic_robust", "logistic_log", "logistic_oversampled")


def baseline_xgb(config):
    c = config.comparison
    return dict(
        n_estimators=c.xgb_estimators,
        max_depth=c.xgb_depth,
        learning_rate=c.xgb_learning_rate,
        subsample=c.xgb_subsample,
        colsample_bytree=c.xgb_colsample,
        reg_lambda=c.xgb_l2,
        class_weight="none",
    )


def suggest(trial, family):
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=10),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 20.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
        }
    if family == "mlp":
        return {
            "width": trial.suggest_categorical("width", [64, 128]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.005, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.01, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024]),
        }
    raise ValueError(f"Unknown search family: {family}")


def make_candidate(family, params, config, *, seed, fit_target):
    c = config.comparison
    if family == "xgboost":
        values = dict(params)
        weighting = values.pop("class_weight")
        if weighting not in {"none", "balanced"}:
            raise ValueError("Unknown XGBoost class weighting.")
        scale = (
            float((fit_target == 0).sum() / (fit_target == 1).sum())
            if weighting == "balanced"
            else 1.0
        )
        return Pipeline(
            [
                (
                    "classifier",
                    XGBClassifier(
                        **values,
                        scale_pos_weight=scale,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        device=config.device,
                        random_state=seed,
                        n_jobs=c.n_jobs,
                    ),
                )
            ]
        )
    if family == "mlp":
        return TorchMLPClassifier(
            **params,
            max_epochs=config.mlp_max_epochs,
            patience=config.mlp_patience,
            inner_folds=config.mlp_inner_folds,
            random_state=seed,
            n_jobs=c.n_jobs,
            device=config.device,
        )
    if family not in ABLATIONS:
        raise ValueError(f"Unknown candidate: {family}")
    steps = []
    if family == "logistic_log":
        steps.append(("signed_log", FunctionTransformer(signed_log1p)))
    steps.extend(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", RobustScaler() if family == "logistic_robust" else StandardScaler()),
        ]
    )
    if family == "logistic_oversampled":
        desired = max(int((fit_target == 1).sum()), int(0.1 * (fit_target == 0).sum()))
        steps.append(
            ("oversampler", RandomOverSampler(sampling_strategy={1: desired}, random_state=seed))
        )
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=c.base.C,
                solver="liblinear",
                random_state=seed,
                max_iter=config.control_max_iter,
                tol=c.base.tol,
            ),
        )
    )
    return (ResamplingPipeline if family == "logistic_oversampled" else Pipeline)(steps)

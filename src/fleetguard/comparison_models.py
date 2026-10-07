"""Fixed-budget candidates; calibration reuses a frozen, already-fitted HGB model."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from fleetguard.comparison_config import ComparisonConfig
from fleetguard.models import make_models


def signed_log1p(values):
    """Stateless heavy-tail compression; defined for negative values and preserves NaNs."""
    return np.sign(values) * np.log1p(np.abs(values))


def make_candidates(config: ComparisonConfig) -> dict:
    models = make_models(config.base)
    for name, indicators, scaler in (
        ("logistic_no_indicator", False, StandardScaler()),
        ("logistic_log_robust", True, RobustScaler()),
    ):
        models[name] = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=indicators, keep_empty_features=True
                    ),
                ),
                ("scaler", scaler),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.base.C,
                        max_iter=config.base.max_iter,
                        tol=config.base.tol,
                        solver="liblinear",
                        random_state=config.base.seed,
                    ),
                ),
            ]
        )
        if name == "logistic_log_robust":
            models[name].steps.insert(0, ("signed_log", FunctionTransformer(signed_log1p)))
    models["random_forest"] = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.forest_estimators,
                    max_depth=config.forest_depth,
                    min_samples_leaf=config.forest_min_leaf,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=config.base.seed,
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )
    # Disable the estimator's internal random holdout: group-aware roles are defined externally.
    models["hist_gradient_boosting"] = Pipeline(
        [
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=config.hgb_iterations,
                    max_leaf_nodes=config.hgb_leaves,
                    learning_rate=config.hgb_learning_rate,
                    l2_regularization=config.hgb_l2,
                    early_stopping=False,
                    random_state=config.base.seed,
                ),
            ),
        ]
    )
    models["xgboost"] = Pipeline(
        [
            (
                "classifier",
                XGBClassifier(
                    n_estimators=config.xgb_estimators,
                    max_depth=config.xgb_depth,
                    learning_rate=config.xgb_learning_rate,
                    subsample=config.xgb_subsample,
                    colsample_bytree=config.xgb_colsample,
                    reg_lambda=config.xgb_l2,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device="cpu",
                    random_state=config.base.seed,
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )
    needed = set(config.models)
    if "hgb_sigmoid" in needed:
        needed.add("hist_gradient_boosting")
    return {name: model for name, model in models.items() if name in needed}


def calibrate_frozen(estimator, features, target):
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(estimator), method="sigmoid", ensemble=False
    )
    calibrated.fit(features, target)
    return calibrated

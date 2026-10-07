"""The first benchmark: two trivial classifiers and two logistic variants."""

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fleetguard.config import Config


def make_models(config: Config) -> dict[str, Pipeline]:
    models = {
        "always_negative": Pipeline(
            [("classifier", DummyClassifier(strategy="constant", constant=0))]
        ),
        "class_prior": Pipeline([("classifier", DummyClassifier(strategy="prior"))]),
    }
    for name, class_weight in (("logistic", None), ("logistic_balanced", "balanced")):
        models[name] = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.C,
                        class_weight=class_weight,
                        solver="liblinear",
                        random_state=config.seed,
                        max_iter=config.max_iter,
                        tol=config.tol,
                    ),
                ),
            ]
        )
    return models

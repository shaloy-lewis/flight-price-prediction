"""Prespecified compact candidate set and realistic non-ML reference."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from src.components.data_transformation import make_preprocessor


class RouteClassMedianRegressor(RegressorMixin, BaseEstimator):
    """Training-only route/cabin median; unknown segments use the training median."""

    def fit(self, X, y):
        frame = X[["source_city", "destination_city", "cabin_class"]].copy()
        frame["target"] = np.asarray(y)
        self.medians_ = (
            frame.groupby(["source_city", "destination_city", "cabin_class"])
            .target.median()
            .to_dict()
        )
        self.global_median_ = float(np.median(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "medians_")
        return np.array(
            [
                self.medians_.get(key, self.global_median_)
                for key in X[["source_city", "destination_city", "cabin_class"]].itertuples(
                    index=False, name=None
                )
            ]
        )


def candidate(name: str, seed: int):
    if name == "route_class_median":
        return RouteClassMedianRegressor(), {}
    base = name.removesuffix("_log")
    models = {
        "median": DummyRegressor(strategy="median"),
        "ridge": Ridge(),
        "tree": DecisionTreeRegressor(random_state=seed, min_samples_leaf=30),
        "hist": HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=120,
            learning_rate=0.08,
            early_stopping=False,
            l2_regularization=1.0,
        ),
    }
    grids: dict[str, dict[str, list[int | float]]] = {
        "median": {},
        "ridge": {"model__alpha": [1.0, 10.0]},
        "tree": {"model__max_depth": [12, 20]},
        "hist": {"model__max_leaf_nodes": [15, 31]},
    }
    pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", models[base])])
    if name.endswith("_log"):
        estimator = TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
        return estimator, {"regressor__" + key: value for key, value in grids[base].items()}
    return pipeline, grids[base]


def fitted_pipeline(estimator):
    return estimator.regressor_ if isinstance(estimator, TransformedTargetRegressor) else estimator


def target_units(estimator) -> str:
    return "log(INR)" if isinstance(estimator, TransformedTargetRegressor) else "INR"

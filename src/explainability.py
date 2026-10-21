"""Attributions use the exact fitted encoded representation and model output scale."""

import numpy as np

from src.components.model_trainer import RouteClassMedianRegressor, fitted_pipeline, target_units


class Explainer:
    def __init__(self, estimator, processed_mean=None):
        self.estimator = estimator
        self.processed_mean = processed_mean
        self.tree_explainer = None

    def explain(self, frame) -> dict:
        if isinstance(self.estimator, RouteClassMedianRegressor):
            prediction = float(self.estimator.predict(frame)[0])
            return {
                "method": "route/class lookup difference from training median",
                "units": "INR",
                "base_value": self.estimator.global_median_,
                "contributions": {"route_class_lookup": prediction - self.estimator.global_median_},
                "model_output": prediction,
                "additivity_error": 0.0,
            }
        pipeline = fitted_pipeline(self.estimator)
        preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
        encoded = preprocessor.transform(frame)
        names = preprocessor.get_feature_names_out().tolist()
        if hasattr(model, "coef_"):
            background = np.asarray(self.processed_mean)
            values = (encoded[0] - background) * model.coef_
            base = float(model.intercept_ + background @ model.coef_)
            method = "exact centered linear contributions (not SHAP)"
        elif hasattr(model, "tree_") or hasattr(model, "_predictors"):
            if self.tree_explainer is None:
                import shap

                self.tree_explainer = shap.TreeExplainer(model)
            values = np.asarray(self.tree_explainer.shap_values(encoded))[0]
            base = float(np.asarray(self.tree_explainer.expected_value).reshape(-1)[0])
            method = "Tree SHAP (tree_path_dependent)"
        else:
            values = np.zeros(len(names))
            base = float(model.predict(encoded)[0])
            method = "constant median baseline"
        if len(names) != len(values):
            raise ValueError("Encoded feature names and attribution dimensions disagree")
        output = float(model.predict(encoded)[0])
        error = float(base + np.sum(values) - output)
        if not np.isclose(base + np.sum(values), output, atol=1e-5, rtol=1e-5):
            raise ValueError("Attribution additivity check failed")
        return {
            "method": method,
            "units": target_units(self.estimator),
            "base_value": base,
            "contributions": dict(zip(names, np.asarray(values, dtype=float).tolist())),
            "model_output": output,
            "additivity_error": error,
        }

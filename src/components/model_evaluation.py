"""Original-currency errors, segment diagnostics and group bootstrap uncertainty."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(y, prediction) -> dict[str, float | None]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty aligned one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or (y <= 0).any():
        raise ValueError("Fares must be positive and all values finite")
    return {
        "mae_inr": float(mean_absolute_error(y, prediction)),
        "rmse_inr": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else None,
        "median_absolute_error_inr": float(median_absolute_error(y, prediction)),
        "mape_percent": float(100 * mean_absolute_percentage_error(y, prediction)),
        "wape_percent": float(100 * np.abs(y - prediction).sum() / y.sum()),
        "bias_inr": float((prediction - y).mean()),
        "underprediction_rate": float((prediction < y).mean()),
    }


def segment_metrics(frame: pd.DataFrame, prediction, price_edges) -> pd.DataFrame:
    scored = frame.copy()
    scored["prediction"] = prediction
    scored["route"] = scored.source_city + " -> " + scored.destination_city
    scored["price_bucket"] = pd.cut(
        scored.price,
        [-np.inf, *price_edges, np.inf],
        labels=["low", "medium", "high", "very_high"],
        duplicates="raise",
    )
    rows = []
    for feature in ("airline", "cabin_class", "route", "stops", "price_bucket"):
        for value, segment in scored.groupby(feature, observed=True):
            rows.append(
                {
                    "feature": feature,
                    "segment": str(value),
                    "n": len(segment),
                    **regression_metrics(segment.price.to_numpy(), segment.prediction.to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap(y, prediction, groups, repetitions: int, seed: int, baseline=None) -> dict:
    """Percentile intervals resample itinerary components, preserving within-group dependence."""
    work = pd.DataFrame({"group": groups, "ae": np.abs(np.asarray(y) - prediction), "n": 1})
    if baseline is not None:
        work["difference"] = work.ae - np.abs(np.asarray(y) - baseline)
    totals = work.groupby("group").sum()
    rng = np.random.default_rng(seed)
    estimates, differences = [], []
    for _ in range(repetitions):
        sampled = totals.iloc[rng.integers(0, len(totals), len(totals))]
        estimates.append(float(sampled.ae.sum() / sampled.n.sum()))
        if baseline is not None:
            differences.append(float(sampled.difference.sum() / sampled.n.sum()))
    result = {
        "method": "itinerary-component percentile bootstrap",
        "confidence": 0.95,
        "repetitions": repetitions,
        "groups": len(totals),
        "mae_inr_interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
    }
    if baseline is not None:
        result["mae_difference_vs_route_class_inr_interval"] = np.quantile(
            differences, [0.025, 0.975]
        ).tolist()
    return result

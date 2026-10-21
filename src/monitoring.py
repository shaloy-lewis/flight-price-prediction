"""Offline drift reference and transparent support warnings; no production claims."""

import numpy as np
import pandas as pd

from src.utils.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_reference(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "categorical": {
            col: frame[col].value_counts(normalize=True).to_dict() for col in CATEGORICAL_FEATURES
        },
        "numeric": {
            col: {
                "minimum": float(frame[col].min()),
                "maximum": float(frame[col].max()),
                "median": float(frame[col].median()),
            }
            for col in NUMERIC_FEATURES
        },
        "routes": sorted(set(frame.source_city + " -> " + frame.destination_city)),
        "airline_cabins": sorted(set(frame.airline + " / " + frame.cabin_class)),
    }


def support_warnings(frame: pd.DataFrame, reference: dict) -> list[str]:
    messages = []
    for col, known in reference["categorical"].items():
        if not frame[col].isin(known).all():
            messages.append(f"{col}: category absent from model training")
    for col, bounds in reference["numeric"].items():
        if ((frame[col] < bounds["minimum"]) | (frame[col] > bounds["maximum"])).any():
            messages.append(f"{col}: outside observed training range")
    if not (frame.source_city + " -> " + frame.destination_city).isin(reference["routes"]).all():
        messages.append("Route absent from model training")
    if not (frame.airline + " / " + frame.cabin_class).isin(reference["airline_cabins"]).all():
        messages.append("Airline/cabin combination absent from model training")
    return messages


def drift_report(frame: pd.DataFrame, reference: dict) -> dict:
    if frame.empty:
        raise ValueError("Cannot compare an empty batch")
    categorical = {}
    for col, known in reference["categorical"].items():
        current = frame[col].value_counts(normalize=True).to_dict()
        categories = set(known) | set(current)
        categorical[col] = {
            "total_variation_distance": float(
                0.5 * sum(abs(known.get(key, 0) - current.get(key, 0)) for key in categories)
            ),
            "unseen_fraction": float((~frame[col].isin(known)).mean()),
        }
    return {
        "rows": len(frame),
        "categorical": categorical,
        "numeric": {
            col: {
                "outside_training_range_fraction": float(
                    ((frame[col] < bounds["minimum"]) | (frame[col] > bounds["maximum"])).mean()
                ),
                "median_shift": float(np.median(frame[col]) - bounds["median"]),
            }
            for col, bounds in reference["numeric"].items()
        },
        "interpretation": "Descriptive drift indicators; not concept drift or measured performance deterioration.",
    }

"""Read original raw Kaggle files, report exclusions, and create leakage groups."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.constants import AIRLINE_CODES, CITIES, FEATURES
from src.utils.utils import (
    categorize_time,
    clock_minutes,
    convert_to_minutes,
    parse_stops,
    parse_travel_date,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Dataset:
    frame: pd.DataFrame
    quality: dict
    sources: list[dict]


def _safe_map(series: pd.Series, parser) -> pd.Series:
    def safe(value):
        try:
            return parser(str(value)) if pd.notna(value) else np.nan
        except (ValueError, TypeError):
            return np.nan

    return series.map({value: safe(value) for value in series.unique()})


def load_dataset(directory: Path | str) -> Dataset:
    directory = Path(directory)
    frames, sources = [], []
    required = {
        "date",
        "airline",
        "ch_code",
        "num_code",
        "dep_time",
        "from",
        "time_taken",
        "stop",
        "arr_time",
        "to",
        "price",
    }
    for cabin in ("economy", "business"):
        path = directory / f"{cabin}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Retrieve the original Kaggle raw files.")
        raw = pd.read_csv(path, dtype=str)
        if not required.issubset(raw.columns):
            raise ValueError(f"{path.name} missing columns: {sorted(required - set(raw.columns))}")
        raw["cabin_class"] = cabin
        raw["source_file"] = path.name
        frames.append(raw)
        sources.append({"file": path.name, "sha256": sha256_file(path), "rows": len(raw)})
    raw = pd.concat(frames, ignore_index=True)
    missing = raw.isna().sum().astype(int).to_dict()
    duplicates = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().reset_index(drop=True)
    frame = raw.rename(columns={"from": "source_city", "to": "destination_city"}).copy()
    for col in ("airline", "source_city", "destination_city", "ch_code", "num_code"):
        frame[col] = frame[col].str.strip()
    frame["departure_time"] = _safe_map(frame.dep_time, categorize_time)
    frame["arrival_time"] = _safe_map(frame.arr_time, categorize_time)
    frame["duration_minutes"] = _safe_map(frame.time_taken, convert_to_minutes)
    frame["stops"] = _safe_map(frame.stop, parse_stops)
    frame["travel_date"] = pd.to_datetime(_safe_map(frame.date, parse_travel_date))
    frame["price"] = pd.to_numeric(
        frame.price.str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )
    reasons = {
        "missing_or_invalid_feature": frame[FEATURES].isna().any(axis=1),
        "invalid_fare": frame.price.isna() | ~np.isfinite(frame.price) | (frame.price <= 0),
        "invalid_travel_date": frame.travel_date.isna(),
        "unknown_airline": ~frame.airline.isin(AIRLINE_CODES),
        "airline_code_mismatch": frame.ch_code != frame.airline.map(AIRLINE_CODES),
        "invalid_route": ~frame.source_city.isin(CITIES)
        | ~frame.destination_city.isin(CITIES)
        | (frame.source_city == frame.destination_city),
        "missing_flight_identifier": frame.num_code.isna() | frame.num_code.eq(""),
    }
    dep = _safe_map(frame.dep_time, clock_minutes)
    arr = _safe_map(frame.arr_time, clock_minutes)
    clock_mismatch = (arr - dep) % 1440 != frame.duration_minutes % 1440
    invalid = pd.DataFrame(reasons).any(axis=1)
    quality = {
        "input_rows": sum(item["rows"] for item in sources),
        "missing_raw": missing,
        "exact_duplicates_removed": duplicates,
        "excluded_rows": int(invalid.sum()),
        "exclusion_reasons_overlapping": {key: int(value.sum()) for key, value in reasons.items()},
        "clock_duration_mismatches_flagged": int(clock_mismatch.sum()),
        "invalid_examples": raw.loc[invalid].head(10).fillna("").to_dict("records"),
    }
    frame = frame.loc[~invalid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows remain; inspect the source schema and quality report")
    frame["flight_id"] = frame.ch_code + "-" + frame.num_code
    quality.update(
        {
            "retained_rows": len(frame),
            "identical_feature_rows": int(frame.duplicated(FEATURES).sum()),
            "feature_groups_with_conflicting_fares": int(
                (frame.groupby(FEATURES).price.nunique() > 1).sum()
            ),
            "travel_date_min": str(frame.travel_date.min().date()),
            "travel_date_max": str(frame.travel_date.max().date()),
            "fare_quantiles_inr": frame.price.quantile([0, 0.01, 0.5, 0.99, 1]).to_dict(),
            "duration_quantiles_minutes": frame.duration_minutes.quantile(
                [0, 0.01, 0.5, 0.99, 1]
            ).to_dict(),
            "observation_timestamp_available": False,
        }
    )
    return Dataset(frame, quality, sources)


def leakage_groups(frame: pd.DataFrame) -> np.ndarray:
    """Join repeated flights AND identical canonical inputs, using no fare information."""
    parents = list(range(len(frame)))

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    for columns in (["airline", "source_city", "destination_city", "flight_id"], FEATURES):
        first = {}
        for idx, key in enumerate(frame[columns].itertuples(index=False, name=None)):
            if key in first:
                a, b = find(idx), find(first[key])
                parents[a] = b
            else:
                first[key] = idx
    return np.array([find(idx) for idx in range(len(frame))])


def holdout_split(frame: pd.DataFrame, groups: np.ndarray, seed: int, test_size: float):
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five independent itinerary groups are needed")
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
            frame, groups=groups
        )
    )
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Group leakage")
    return train, test

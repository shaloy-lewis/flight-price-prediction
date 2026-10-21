"""Small deterministic synthetic fixtures. Never portfolio performance evidence."""

import pandas as pd
import pytest

from src.artifacts import save_artifact
from src.components.model_trainer import candidate, fitted_pipeline, target_units
from src.monitoring import make_reference
from src.utils.constants import FEATURES


@pytest.fixture
def fixture_frame():
    rows = []
    for idx in range(80):
        rows.append(
            {
                "airline": ["SpiceJet", "Air India"][idx % 2],
                "source_city": ["Delhi", "Mumbai"][idx % 2],
                "destination_city": ["Mumbai", "Delhi"][idx % 2],
                "cabin_class": ["economy", "business"][(idx // 2) % 2],
                "departure_time": "Night",
                "arrival_time": "Late_Night",
                "duration_minutes": float(120 + idx * 5),
                "stops": int(idx % 3),
                "flight_id": f"synthetic-{idx // 2}",
                "price": float(2500 + idx * 100),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_artifact(tmp_path, fixture_frame):
    model, _ = candidate("tree_log", 42)
    model.set_params(regressor__model__min_samples_leaf=2)
    model.fit(fixture_frame[FEATURES], fixture_frame.price)
    pipeline = fitted_pipeline(model)
    mean = (
        pipeline.named_steps["preprocessor"]
        .transform(fixture_frame[FEATURES])
        .mean(axis=0)
        .tolist()
    )
    directory = tmp_path / "synthetic-model"
    save_artifact(
        directory,
        model,
        {
            "model_version": "synthetic-test-only",
            "selected_model": "tree_log",
            "reference": make_reference(fixture_frame[FEATURES]),
            "target_units": target_units(model),
            "data_sources": [{"file": "synthetic fixture", "sha256": "not-real-data"}],
            "processed_training_mean": mean,
        },
    )
    return directory


@pytest.fixture
def request_payload(fixture_frame):
    return fixture_frame[FEATURES].iloc[0].to_dict()


@pytest.fixture(autouse=True)
def single_thread_numeric_work():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield

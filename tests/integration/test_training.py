import importlib
import json

import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.pipeline.prediction_pipeline import PredictPipeline
from src.pipeline.training_pipeline import run_training
from src.utils.constants import FEATURES


def test_training_import_has_no_side_effects(tmp_path, monkeypatch):
    import src.pipeline.training_pipeline as module

    monkeypatch.chdir(tmp_path)
    importlib.reload(module)
    assert not list(tmp_path.iterdir())


def test_end_to_end_synthetic_training(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for cabin in ("economy", "business"):
        rows = []
        for idx in range(36):
            duration = 60 + idx * 5
            arrival = 480 + duration
            rows.append(
                dict(
                    date="11-02-2022",
                    airline="Air India",
                    ch_code="AI",
                    num_code=str(100 + idx),
                    dep_time="08:00",
                    **{"from": "Delhi", "to": "Mumbai"},
                    time_taken=f"{duration // 60}h {duration % 60}m",
                    stop="non-stop",
                    arr_time=f"{arrival // 60:02d}:{arrival % 60:02d}",
                    price=str((2000 if cabin == "economy" else 20000) + idx * 100),
                )
            )
        pd.DataFrame(rows).to_csv(data / f"{cabin}.csv", index=False)
    output = tmp_path / "synthetic-run"
    config = ExperimentConfig(
        models=("median", "route_class_median", "ridge", "tree", "tree_log"),
        bootstrap_repetitions=20,
        permutation_sample=10,
        permutation_repeats=2,
    )
    result = run_training(data, output, config)
    assert result["train_groups"] + result["holdout_groups"] == result["total_groups"]
    assert (output / "manifest.json").is_file()
    assert len(result["comparisons"]) == 5
    assert (
        json.loads((output / "selection.json").read_text())["holdout_used_for_selection"] is False
    )
    from src.components.data_ingestion import load_dataset

    frame = load_dataset(data).frame
    predictor = PredictPipeline(output)
    assert predictor.predict(frame[FEATURES].iloc[:2]).shape == (2,)
    with pytest.raises(ValueError, match="empty"):
        run_training(data, output, config)

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.components.data_ingestion import holdout_split, leakage_groups, load_dataset
from src.components.model_evaluation import cluster_bootstrap, regression_metrics
from src.components.model_trainer import candidate
from src.config import ExperimentConfig
from src.monitoring import drift_report, make_reference
from src.utils.constants import FEATURES


def test_no_flight_or_feature_leakage_and_no_target_use(fixture_frame):
    frame = pd.concat(
        [fixture_frame, fixture_frame.iloc[:8].assign(flight_id="another-flight")],
        ignore_index=True,
    )
    groups = leakage_groups(frame)
    np.testing.assert_array_equal(groups, leakage_groups(frame.assign(price=1)))
    train, test = holdout_split(frame, groups, 42, 0.2)
    np.testing.assert_array_equal(train, holdout_split(frame, groups, 42, 0.2)[0])
    assert not set(groups[train]) & set(groups[test])

    def keys(indices, columns):
        return set(frame.iloc[indices][columns].itertuples(index=False, name=None))

    assert not keys(train, FEATURES) & keys(test, FEATURES)
    assert not keys(train, ["airline", "source_city", "destination_city", "flight_id"]) & keys(
        test, ["airline", "source_city", "destination_city", "flight_id"]
    )
    for a, b in GroupKFold(3).split(frame, groups=groups):
        assert not set(groups[a]) & set(groups[b])


def test_data_quality_and_quarantine(tmp_path):
    raw = dict(
        date="11-02-2022",
        airline="SpiceJet",
        ch_code="SG",
        num_code="8709",
        dep_time="18:55",
        **{"from": "Delhi", "to": "Mumbai"},
        time_taken="02h 10m",
        stop="non-stop",
        arr_time="21:05",
        price="5,953",
    )
    rows = [
        raw,
        raw,
        {**raw, "time_taken": "1.03h m"},
        {**raw, "price": "0"},
        {**raw, "ch_code": "AI"},
        {**raw, "date": "31-02-2022"},
        {**raw, "stop": "11-stop"},
        {**raw, "num_code": None},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "economy.csv", index=False)
    pd.DataFrame([raw]).to_csv(tmp_path / "business.csv", index=False)
    dataset = load_dataset(tmp_path)
    assert dataset.quality["exact_duplicates_removed"] == 1
    assert dataset.quality["excluded_rows"] == 6
    assert len(dataset.frame) == 2
    assert (dataset.frame.price == 5953).all()


def test_missing_data_does_not_train(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "name", ["median", "route_class_median", "ridge", "tree", "tree_log", "hist", "hist_log"]
)
def test_candidate_shape_determinism_and_inverse_transform(fixture_frame, name):
    model, _ = candidate(name, 42)
    X, y = fixture_frame[FEATURES], fixture_frame.price.to_numpy()
    model.fit(X, y)
    first = model.predict(X)
    second = clone(model).fit(X, y).predict(X)
    assert first.shape == y.shape
    assert np.isfinite(first).all() and (first > 0).all()
    np.testing.assert_allclose(first, second)


def test_metrics_and_bootstrap():
    y, pred = np.array([100.0, 200.0, 300.0]), np.array([90.0, 210.0, 300.0])
    metrics = regression_metrics(y, pred)
    assert metrics["mae_inr"] == pytest.approx(20 / 3)
    assert metrics["rmse_inr"] == pytest.approx(np.sqrt(200 / 3))
    result = cluster_bootstrap(y, pred, [1, 1, 2], 50, 42, pred + 5)
    assert result == cluster_bootstrap(y, pred, [1, 1, 2], 50, 42, pred + 5)
    assert result["groups"] == 2


def test_monitoring_descriptive(fixture_frame):
    reference = make_reference(fixture_frame)
    report = drift_report(fixture_frame, reference)
    assert report["categorical"]["airline"]["total_variation_distance"] == 0
    changed = fixture_frame.assign(airline="UNSEEN", duration_minutes=5000)
    drift = drift_report(changed, reference)
    assert drift["categorical"]["airline"]["unseen_fraction"] == 1
    assert drift["numeric"]["duration_minutes"]["outside_training_range_fraction"] == 1


def test_config_rejects_invalid():
    with pytest.raises(ValueError):
        ExperimentConfig(cv_folds=1)
    with pytest.raises(ValueError):
        ExperimentConfig(models=("tree",))

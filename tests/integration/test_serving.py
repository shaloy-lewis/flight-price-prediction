import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import create_app
from src.artifacts import ArtifactError, load_artifact
from src.components.model_trainer import candidate, fitted_pipeline
from src.explainability import Explainer
from src.pipeline.prediction_pipeline import PredictPipeline
from src.utils.constants import FEATURES


def test_artifact_prediction_consistency_and_shap(trained_artifact, fixture_frame):
    pipeline = PredictPipeline(trained_artifact)
    features = fixture_frame[FEATURES].iloc[:4].copy()
    original = features.copy(deep=True)
    expected = pipeline.model.predict(features)
    np.testing.assert_allclose(pipeline.predict(features), expected)
    pd.testing.assert_frame_equal(features, original)
    explanation = pipeline.explain(features.iloc[:1])
    encoded_names = (
        fitted_pipeline(pipeline.model).named_steps["preprocessor"].get_feature_names_out()
    )
    assert set(explanation["contributions"]) == set(encoded_names)
    assert explanation["units"] == "log(INR)"
    assert explanation["base_value"] + sum(explanation["contributions"].values()) == pytest.approx(
        explanation["model_output"]
    )
    assert np.exp(explanation["model_output"]) == pytest.approx(expected[0])
    assert explanation["additivity_error"] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    "name", ["ridge", "tree", "hist", "hist_log", "median", "route_class_median"]
)
def test_explanation_for_each_model(fixture_frame, name):
    estimator, _ = candidate(name, 42)
    estimator.fit(fixture_frame[FEATURES], fixture_frame.price)
    pipeline = fitted_pipeline(estimator)
    mean = None
    if hasattr(pipeline, "named_steps"):
        mean = pipeline.named_steps["preprocessor"].transform(fixture_frame[FEATURES]).mean(axis=0)
    explanation = Explainer(estimator, mean).explain(fixture_frame[FEATURES].iloc[:1])
    assert explanation["base_value"] + sum(explanation["contributions"].values()) == pytest.approx(
        explanation["model_output"]
    )
    output = explanation["model_output"]
    if name.endswith("_log"):
        output = np.exp(output)
    assert output == pytest.approx(estimator.predict(fixture_frame[FEATURES].iloc[:1])[0])


def test_untrusted_corrupt_and_mismatched_artifacts(trained_artifact, monkeypatch):
    with pytest.raises(ArtifactError, match="trusted"):
        load_artifact(trained_artifact)
    manifest_path = trained_artifact / "manifest.json"
    original = json.loads(manifest_path.read_text())
    for key, value in [("schema_version", "0"), ("packages", {}), ("python", "0.0")]:
        manifest_path.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ArtifactError):
            load_artifact(trained_artifact, trusted=True)
    manifest_path.write_text(json.dumps(original))
    (trained_artifact / "model.joblib").write_bytes(b"corrupted")
    import src.artifacts

    def never_load(*args, **kwargs):
        pytest.fail("Deserialization must not run after integrity failure")

    monkeypatch.setattr(src.artifacts.joblib, "load", never_load)
    with pytest.raises(ArtifactError, match="integrity"):
        load_artifact(trained_artifact, trusted=True)


def test_api_success_contract_and_validation(trained_artifact, request_payload):
    with TestClient(create_app(trained_artifact)) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/version").json()["schema_version"] == "2.0"
        response = client.post("/predict?explain=true", json=request_payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["flight_price_prediction"] > 0
        assert body["currency"] == "INR"
        assert body["explanation"]["units"] == "log(INR)"
        assert body["model_version"] == "synthetic-test-only"
        assert client.post("/predict", json={}).status_code == 422
        assert client.post("/predict", json={**request_payload, "stops": 3}).status_code == 422
        assert (
            client.post("/predict", json={**request_payload, "duration_minutes": -1}).status_code
            == 422
        )
        assert (
            client.post("/predict", json={**request_payload, "airline": "UNKNOWN"}).status_code
            == 422
        )


def test_api_legacy(trained_artifact):
    payload = dict(
        airline="SpiceJet",
        from_="Delhi",
        to_="Mumbai",
        class_="economy",
        ch_code="SG",
        dep_time="23:00",
        arr_time="01:00",
        date="11-02-2022",
        time_taken="02h",
        stop="non-stop",
    )
    with TestClient(create_app(trained_artifact)) as client:
        first = client.post("/predict/legacy", json=payload)
        second = client.post("/predict/legacy", json={**payload, "date": "11-02-2030"})
        assert first.status_code == 200
        assert first.json()["flight_price_prediction"] == second.json()["flight_price_prediction"]
        assert any("ignored" in item for item in first.json()["warnings"])
        assert client.post("/predict/legacy", json={**payload, "ch_code": "AI"}).status_code == 422


def test_api_unavailable_and_internal_failure(
    tmp_path, trained_artifact, request_payload, monkeypatch
):
    with TestClient(create_app(tmp_path / "missing")) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=request_payload).status_code == 503
    predictor = PredictPipeline(trained_artifact)

    def fail(*args, **kwargs):
        raise RuntimeError("secret/path/internal")

    monkeypatch.setattr(predictor, "predict", fail)
    with TestClient(create_app(predictor=predictor)) as client:
        response = client.post("/predict", json=request_payload)
        assert response.status_code == 500
        assert "secret/path" not in response.text


def test_pipeline_invalid_outputs(trained_artifact, request_payload, monkeypatch):
    pipeline = PredictPipeline(trained_artifact)
    monkeypatch.setattr(pipeline.model, "predict", lambda frame: np.array([np.nan]))
    with pytest.raises(RuntimeError, match="invalid"):
        pipeline.predict(pd.DataFrame([request_payload]))
    with pytest.raises(ValueError):
        pipeline.predict(pd.DataFrame())


def test_model_directory_environment_and_explicit_override(
    trained_artifact, request_payload, monkeypatch
):
    monkeypatch.setenv("MODEL_DIR", str(trained_artifact))
    from_environment = PredictPipeline()
    monkeypatch.setenv("MODEL_DIR", str(trained_artifact / "missing"))
    explicit = PredictPipeline(trained_artifact)
    frame = pd.DataFrame([request_payload])
    np.testing.assert_allclose(from_environment.predict(frame), explicit.predict(frame))


def test_empty_model_directory_uses_default(trained_artifact, request_payload, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "")
    # Put the trusted fixture at the documented default relative location.
    monkeypatch.chdir(trained_artifact.parent)
    import shutil

    default = trained_artifact.parent / "artifacts" / "v2"
    shutil.copytree(trained_artifact, default)
    pipeline = PredictPipeline()
    assert pipeline.predict(pd.DataFrame([request_payload])).shape == (1,)

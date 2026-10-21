"""Load once, validate every request, and share the fitted training pipeline."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import load_artifact
from src.explainability import Explainer
from src.monitoring import support_warnings
from src.schema import FlightRequest, LegacyFlightRequest
from src.utils.constants import FEATURES


class PredictPipeline:
    def __init__(self, artifact_dir: Path | str | None = None):
        directory = (
            Path(artifact_dir) if artifact_dir else Path(os.getenv("MODEL_DIR") or "artifacts/v2")
        )
        self.model, self.manifest = load_artifact(directory, trusted=True)
        self.explainer = Explainer(self.model, self.manifest.get("processed_training_mean"))

    def validated_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            raise ValueError("At least one itinerary is required")
        records = [
            FlightRequest.model_validate(row).model_dump() for row in features.to_dict("records")
        ]
        return pd.DataFrame(records, columns=FEATURES)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        frame = self.validated_frame(features)
        result = np.asarray(self.model.predict(frame), dtype=float)
        if result.shape != (len(frame),) or not np.isfinite(result).all() or (result <= 0).any():
            raise RuntimeError("Model produced invalid fare estimates")
        return result

    def warnings(self, features: pd.DataFrame) -> list[str]:
        return support_warnings(self.validated_frame(features), self.manifest["reference"])

    def explain(self, features: pd.DataFrame) -> dict:
        frame = self.validated_frame(features)
        if len(frame) != 1:
            raise ValueError("Local explanation requires exactly one itinerary")
        return self.explainer.explain(frame)


class CustomData:
    """Compatibility adapter for original Python clients; date is validated and discarded."""

    def __init__(self, **kwargs):
        self.request = LegacyFlightRequest(**kwargs)

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.request.to_canonical().model_dump()], columns=FEATURES)

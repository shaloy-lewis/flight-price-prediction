"""FastAPI serving with one trusted model per process and explicit historical scope."""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.artifacts import ArtifactError
from src.pipeline.prediction_pipeline import PredictPipeline
from src.schema import FlightRequest, LegacyFlightRequest
from src.utils.constants import PROJECT_VERSION, SCHEMA_VERSION, SCOPE

logger = logging.getLogger("fare_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.propagate = False


class ExplanationResponse(BaseModel):
    method: str
    units: str
    base_value: float
    contributions: dict[str, float]
    model_output: float
    additivity_error: float


class PredictionResponse(BaseModel):
    flight_price_prediction: float
    currency: str = "INR"
    model_version: str
    schema_version: str = SCHEMA_VERSION
    scope: str = SCOPE
    warnings: list[str]
    uncertainty: str = "No calibrated individual prediction interval is available."
    explanation: ExplanationResponse | None = None


def create_app(artifact_dir: Path | str | None = None, predictor=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor
        if predictor is None:
            try:
                application.state.predictor = PredictPipeline(artifact_dir)
            except ArtifactError:
                logger.exception("Model unavailable at initialization")
        yield

    application = FastAPI(
        title="Historical Flight Fare Benchmark",
        version=PROJECT_VERSION,
        description=SCOPE,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )
        )
        return response

    @application.get("/")
    def root():
        return {"message": SCOPE, "documentation": "/docs"}

    @application.get("/ping")
    def ping():
        return {"status": "alive"}

    @application.get("/health")
    def health():
        if application.state.predictor is None:
            raise HTTPException(
                503, "Model unavailable. Train or mount a compatible versioned artifact."
            )
        return {
            "status": "ready",
            "model_version": application.state.predictor.manifest["model_version"],
        }

    @application.get("/version")
    def version():
        model = application.state.predictor
        return {
            "application_version": PROJECT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "model_version": model.manifest["model_version"] if model else None,
        }

    def respond(data: FlightRequest, explain: bool) -> PredictionResponse:
        model = application.state.predictor
        if model is None:
            raise HTTPException(
                503, "Model unavailable. Train or mount a compatible versioned artifact."
            )
        frame = pd.DataFrame([data.model_dump()])
        try:
            prediction = float(model.predict(frame)[0])
            explanation = model.explain(frame) if explain else None
            return PredictionResponse(
                flight_price_prediction=round(prediction, 2),
                model_version=model.manifest["model_version"],
                warnings=[SCOPE, *model.warnings(frame)],
                explanation=explanation,
            )
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(500, "Prediction failed; consult server logs.") from exc

    @application.post("/predict", response_model=PredictionResponse)
    def predict(data: FlightRequest, explain: bool = False):
        return respond(data, explain)

    @application.post("/predict/legacy", response_model=PredictionResponse, deprecated=True)
    def legacy(data: LegacyFlightRequest, explain: bool = False):
        try:
            canonical = data.to_canonical()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        response = respond(canonical, explain)
        response.warnings.append(
            "Legacy travel date was validated and ignored; booking lead time is not modeled."
        )
        return response

    return application


app = create_app()

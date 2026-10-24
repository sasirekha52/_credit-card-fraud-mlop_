import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import load_model
from api.schemas import HealthResponse, PredictRequest, PredictResponse

NOTE = (
    "This is a risk signal from a statistical model, not proof of fraud. "
    "A flagged transaction should be reviewed by a person."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model a single time.
    app.state.model, app.state.info, app.state.error = None, None, None
    try:
        app.state.model, app.state.info = load_model()
    except Exception as exc:  # keep the API alive so /health can report the problem
        app.state.error = str(exc)
    yield


app = FastAPI(title="Credit Card Fraud Risk API", version="0.1.0", lifespan=lifespan)


def risk_level(probability: float, threshold: float) -> str:
    """Simple demo rule: high at/above the threshold, medium above half of it."""
    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.5:
        return "medium"
    return "low"


@app.get("/health", response_model=HealthResponse)
def health():
    if app.state.model is None:
        return HealthResponse(status="degraded", model_loaded=False, detail=app.state.error)
    info = app.state.info
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=info["model_name"],
        model_version=info["model_version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    info = app.state.info
    start = time.perf_counter()
    # Column order must match training exactly, so we use the saved feature list.
    row = pd.DataFrame([request.features.model_dump()])[info["features"]]
    probability = float(app.state.model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    threshold = info["threshold"]
    flagged = probability >= threshold
    return PredictResponse(
        request_id=str(uuid.uuid4()),
        transaction_id=request.transaction_id,
        timestamp=datetime.now(timezone.utc),
        fraud_probability=round(probability, 6),
        threshold=round(threshold, 4),
        decision="flag_for_review" if flagged else "allow",
        risk_level=risk_level(probability, threshold),
        model_name=info["model_name"],
        model_version=info["model_version"],
        latency_ms=round(latency_ms, 2),
        note=NOTE,
    )
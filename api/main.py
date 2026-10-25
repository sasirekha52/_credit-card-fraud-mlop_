import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from api.db import SessionLocal
from api.db_models import Prediction, PredictionFeatures
from api.model_loader import load_model
from api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    TransactionOut,
    TransactionsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

NOTE = (
    "This is a risk signal from a statistical model, not proof of fraud. "
    "A flagged transaction should be reviewed by a person."
)

_counter_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.info, app.state.error = None, None, None
    app.state.started_at = time.time()
    app.state.predictions = 0
    app.state.db_failures = 0
    try:
        app.state.model, app.state.info = load_model()
    except Exception as exc:  # keep the API alive so /health can report the problem
        app.state.error = str(exc)
    yield


app = FastAPI(title="Credit Card Fraud Risk API", version="0.2.0", lifespan=lifespan)


def risk_level(probability: float, threshold: float) -> str:
    """Simple demo rule: high at/above the threshold, medium above half of it."""
    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.5:
        return "medium"
    return "low"


def database_ok() -> bool:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def save_prediction(response: PredictResponse, features: dict) -> None:
    """Runs after the response is sent. A failure here must never affect the caller."""
    try:
        with SessionLocal() as session:
            session.add(
                Prediction(
                    request_id=response.request_id,
                    transaction_id=response.transaction_id,
                    created_at=response.timestamp,
                    fraud_probability=response.fraud_probability,
                    threshold=response.threshold,
                    decision=response.decision,
                    risk_level=response.risk_level,
                    model_name=response.model_name,
                    model_version=response.model_version,
                    latency_ms=response.latency_ms,
                    amount=features["Amount"],
                    source=response.source,
                )
            )
            session.add(PredictionFeatures(request_id=response.request_id, features=features))
            session.commit()
    except SQLAlchemyError as exc:
        with _counter_lock:
            app.state.db_failures += 1
        # Log only the error TYPE: full messages can contain the stored values.
        logger.error("Could not save prediction %s (%s)", response.request_id, type(exc).__name__)


@app.get("/health", response_model=HealthResponse)
def health():
    db = "ok" if database_ok() else "unavailable"
    if app.state.model is None:
        return HealthResponse(
            status="degraded", model_loaded=False, database=db, detail=app.state.error
        )
    info = app.state.info
    return HealthResponse(
        status="ok" if db == "ok" else "degraded",
        model_loaded=True,
        database=db,
        model_name=info["model_name"],
        model_version=info["model_version"],
        detail=None if db == "ok" else "Predictions work but are not being logged.",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    info = app.state.info
    features = request.features.model_dump()
    start = time.perf_counter()
    # Column order must match training exactly, so we use the saved feature list.
    row = pd.DataFrame([features])[info["features"]]
    probability = float(app.state.model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    threshold = info["threshold"]
    flagged = probability >= threshold
    response = PredictResponse(
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
        source=request.source,
        note=NOTE,
    )
    with _counter_lock:
        app.state.predictions += 1
    background_tasks.add_task(save_prediction, response, features)
    return response


@app.get("/transactions", response_model=TransactionsResponse)
def transactions(
    limit: int = Query(50, ge=1, le=500),
    decision: Literal["flag_for_review", "allow"] | None = None,
    source: Literal["api", "simulated"] | None = None,
):
    try:
        with SessionLocal() as session:
            query = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
            if decision:
                query = query.where(Prediction.decision == decision)
            if source:
                query = query.where(Prediction.source == source)
            rows = session.scalars(query).all()
            items = [TransactionOut.model_validate(r) for r in rows]
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database unavailable. Try again later.")
    return TransactionsResponse(count=len(items), transactions=items)


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    total = flagged = avg_ms = p95_ms = None
    db_up = True
    try:
        with SessionLocal() as session:
            total, flagged, avg_ms, p95_ms = session.execute(
                select(
                    func.count(Prediction.request_id),
                    func.count(Prediction.request_id).filter(
                        Prediction.decision == "flag_for_review"
                    ),
                    func.avg(Prediction.latency_ms),
                    func.percentile_cont(0.95).within_group(Prediction.latency_ms),
                )
            ).one()
    except SQLAlchemyError:
        db_up = False

    return MetricsResponse(
        uptime_seconds=round(time.time() - app.state.started_at, 1),
        predictions_this_session=app.state.predictions,
        db_write_failures=app.state.db_failures,
        database_available=db_up,
        total_logged=total,
        flagged_for_review=flagged,
        avg_latency_ms=round(float(avg_ms), 2) if avg_ms is not None else None,
        p95_latency_ms=round(float(p95_ms), 2) if p95_ms is not None else None,
    )
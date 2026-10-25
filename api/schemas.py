from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

# One float field per model feature: V1..V28 and Amount.
_fields = {f"V{i}": (float, Field(..., allow_inf_nan=False)) for i in range(1, 29)}
_fields["Amount"] = (float, Field(..., ge=0, allow_inf_nan=False))  # cannot be negative

TransactionFeatures = create_model("TransactionFeatures", **_fields)


class PredictRequest(BaseModel):
    transaction_id: str | None = Field(default=None, max_length=64)
    source: Literal["api", "simulated"] = "api"
    features: TransactionFeatures


class PredictResponse(BaseModel):
    request_id: str
    transaction_id: str | None
    timestamp: datetime
    fraud_probability: float
    threshold: float
    decision: Literal["flag_for_review", "allow"]
    risk_level: Literal["low", "medium", "high"]
    model_name: str
    model_version: str
    latency_ms: float
    source: Literal["api", "simulated"]
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database: Literal["ok", "unavailable"]
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    transaction_id: str | None
    created_at: datetime
    fraud_probability: float
    threshold: float
    decision: str
    risk_level: str
    model_version: str
    latency_ms: float
    amount: float
    source: str
    actual_label: int | None  # None = outcome not known yet


class TransactionsResponse(BaseModel):
    count: int
    transactions: list[TransactionOut]


class MetricsResponse(BaseModel):
    uptime_seconds: float
    predictions_this_session: int
    db_write_failures: int
    database_available: bool
    total_logged: int | None
    flagged_for_review: int | None
    avg_latency_ms: float | None
    p95_latency_ms: float | None
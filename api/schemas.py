from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, create_model

# One float field per model feature: V1..V28 and Amount.
# create_model builds the class from a loop so we don't write 29 lines by hand.
_fields = {f"V{i}": (float, Field(..., allow_inf_nan=False)) for i in range(1, 29)}
_fields["Amount"] = (float, Field(..., ge=0, allow_inf_nan=False))  # amount cannot be negative

TransactionFeatures = create_model("TransactionFeatures", **_fields)


class PredictRequest(BaseModel):
    transaction_id: str | None = Field(default=None, max_length=64)
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
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None
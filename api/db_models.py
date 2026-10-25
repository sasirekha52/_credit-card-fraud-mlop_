from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    transaction_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="api")
    # Ground truth: stays NULL until the real outcome is known (0 = legit, 1 = fraud)
    actual_label: Mapped[int | None] = mapped_column(Integer)
    label_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_predictions_created_at", "created_at"),)


class PredictionFeatures(Base):
    __tablename__ = "prediction_features"

    request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.request_id", ondelete="CASCADE"), primary_key=True
    )
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import engine
from api.db_models import Prediction, PredictionFeatures


def test_insert_both_tables_then_roll_back():
    try:
        conn = engine.connect()
    except SQLAlchemyError:
        pytest.skip("PostgreSQL is not available")
    request_id = str(uuid.uuid4())
    transaction = conn.begin()
    session = Session(bind=conn)
    try:
        session.add(Prediction(
            request_id=request_id, transaction_id="pytest-db", fraud_probability=0.5,
            threshold=0.3649, decision="flag_for_review", risk_level="high",
            model_name="fraud-detector", model_version="1", latency_ms=1.0,
            amount=10.0, source="simulated",
        ))
        session.add(PredictionFeatures(request_id=request_id, features={"V1": 0.0}))
        session.flush()
        count = session.scalar(
            select(func.count()).select_from(PredictionFeatures)
            .where(PredictionFeatures.request_id == request_id)
        )
        assert count == 1
    finally:
        session.close()
        transaction.rollback()  # nothing is left in the real database
        conn.close()
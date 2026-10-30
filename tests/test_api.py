import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import api.main as main

ORIGINAL_SAVE = main.save_prediction
FEATURES = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0}


def payload(**overrides):
    return {"transaction_id": "pytest-1", "source": "simulated",
            "features": {**FEATURES, **overrides}}


def failing_session():
    raise OperationalError("SELECT 1", {}, Exception("simulated outage"))


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:  # runs startup: loads the model once
        yield c


@pytest.fixture(autouse=True)
def no_real_db_writes(monkeypatch):
    monkeypatch.setattr(main, "save_prediction", lambda *args, **kwargs: None)


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_expected_fields(client):
    resp = client.post("/predict", json=payload())
    assert resp.status_code == 200
    body = resp.json()
    for key in ["request_id", "timestamp", "fraud_probability", "threshold", "decision",
                "risk_level", "model_version", "latency_ms", "source", "note"]:
        assert key in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["source"] == "simulated"
    assert "not proof of fraud" in body["note"]


def test_decision_matches_threshold(client):
    body = client.post("/predict", json=payload()).json()
    assert (body["decision"] == "flag_for_review") == (body["fraud_probability"] >= body["threshold"])


def test_missing_field_is_rejected(client):
    bad = payload()
    del bad["features"]["V5"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
    assert "V5" in str(resp.json()["detail"])


def test_negative_amount_is_rejected(client):
    assert client.post("/predict", json=payload(Amount=-5)).status_code == 422


def test_non_numeric_value_is_rejected(client):
    assert client.post("/predict", json=payload(V1="abc")).status_code == 422


def test_predict_returns_503_when_model_missing(client):
    saved = client.app.state.model
    client.app.state.model = None
    try:
        resp = client.post("/predict", json=payload())
    finally:
        client.app.state.model = saved
    assert resp.status_code == 503


def test_predict_survives_database_failure(client, monkeypatch):
    monkeypatch.setattr(main, "save_prediction", ORIGINAL_SAVE)
    monkeypatch.setattr(main, "SessionLocal", failing_session)
    before = client.app.state.db_failures
    resp = client.post("/predict", json=payload())
    assert resp.status_code == 200
    assert client.app.state.db_failures == before + 1


def test_transactions_returns_503_when_database_down(client, monkeypatch):
    monkeypatch.setattr(main, "SessionLocal", failing_session)
    assert client.get("/transactions").status_code == 503
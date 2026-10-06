"""Integration and contract tests for FastAPI service."""

import json

import pytest
from fastapi.testclient import TestClient

from fraudguard.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_version" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "raw_predictors" in data
    assert len(data["raw_predictors"]) == 12


def test_score_valid_payload(client):
    with open("examples/synthetic_transaction.json", "r") as f:
        payload = json.load(f)

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["TransactionID"] == payload["TransactionID"]
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert isinstance(data["review_recommended"], bool)
    assert data["score_type"] in ["raw_probability", "calibrated_probability"]


def test_score_extra_field_rejected(client):
    with open("examples/synthetic_transaction.json", "r") as f:
        payload = json.load(f)
    payload["extra_field"] = "malicious_or_unknown"

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422


def test_score_isFraud_field_rejected(client):
    with open("examples/synthetic_transaction.json", "r") as f:
        payload = json.load(f)
    payload["isFraud"] = 1

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422


def test_score_negative_amount_rejected(client):
    with open("examples/synthetic_transaction.json", "r") as f:
        payload = json.load(f)
    payload["TransactionAmt"] = -10.0

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422


def test_score_oversized_payload_rejected(client):
    with open("examples/synthetic_transaction.json", "r") as f:
        payload = json.load(f)
    payload["P_emaildomain"] = "x" * 70000  # exceeds 64 KiB

    response = client.post("/v1/score", json=payload)
    assert response.status_code in [413, 422]

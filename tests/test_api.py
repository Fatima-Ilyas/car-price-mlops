# API tests: do the API's "doors" work?

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
CAR = {"age_years": 5, "mileage_thousand_km": 60, "engine_cc": 1300}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_price():
    response = client.post("/predict", json=CAR)
    assert response.status_code == 200
    assert response.json()["price_lakh"] > 0


def test_bad_input_is_rejected():
    response = client.post("/predict", json={**CAR, "age_years": "five"})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text

from fastapi.testclient import TestClient
from src.app import app
import time

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()

def test_predict_endpoint():
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0

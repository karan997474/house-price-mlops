from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "House Price Prediction API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "area": 2100,
            "bedrooms": 4,
            "age": 3
        }
    )

    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert response.json()["predicted_price"] > 0

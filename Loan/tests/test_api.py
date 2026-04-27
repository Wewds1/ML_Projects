from fastapi.testclient import TestClient

from src.loan_predictor.api import app


def test_root_route():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

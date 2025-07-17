from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "MarketSense AI Backend" in response.json().get("message", "")

def test_lead_score_predict():
    payload = {
        "engagement_score": 50,
        "time_spent": 30.0,
        "source": "Social Media",
        "industry": "Tech"
    }
    response = client.post("/predict-lead-score", json=payload)
    assert response.status_code == 200
    assert "converted_prediction" in response.json()
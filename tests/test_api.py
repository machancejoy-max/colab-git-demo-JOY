import json
from app import app

def test_predict_positive():
    client = app.test_client()
    response = client.post(
        "/predict",
        data=json.dumps({"text": "I love this!"}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data

def test_predict_negative():
    client = app.test_client()
    response = client.post(
        "/predict",
        data=json.dumps({"text": "This is terrible"}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data

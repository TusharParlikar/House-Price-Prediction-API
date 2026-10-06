from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# First row of the California Housing dataset (actual value: $452,600)
HOUSE = {
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
}


def test_home():
    assert client.get("/").status_code == 200


def test_predict():
    r = client.post("/predict", json=HOUSE)
    assert r.status_code == 200
    assert 300_000 < r.json()["predicted_price"] < 600_000


def test_predict_batch_matches_single():
    r = client.post("/predict_batch", json={"houses": [HOUSE, HOUSE]})
    assert r.status_code == 200
    single = client.post("/predict", json=HOUSE).json()["predicted_price"]
    assert r.json()["predicted_prices"] == [single, single]

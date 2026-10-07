import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# First row of the California Housing dataset (actual value: $452,600)
HOUSE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def test_predict_and_batch_agree():
    single = client.post("/predict", json=HOUSE)
    assert single.status_code == 200
    price = single.json()["predicted_price"]
    assert 300_000 < price < 600_000

    batch = client.post("/predict_batch", json={"houses": [HOUSE, HOUSE]})
    assert batch.json()["predicted_prices"] == [price, price]


def test_cors_lets_browser_apps_call_api():
    # Preflight a browser (e.g. test/index.html) sends before a cross-origin JSON POST
    r = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize(
    "change",
    [
        {"population": -1},
        {"population": float("nan")},
        {"households": 0},  # the model divides by households
        {"ocean_proximity": "BEACH"},
        {"longitude": -80.0},  # outside California
        {"total_rooms": 100, "total_bedrooms": 500},  # more bedrooms than rooms
    ],
)
def test_predict_rejects_invalid_input(change):
    # json.dumps writes NaN as a token, which the API's JSON parser accepts
    body = json.dumps({**HOUSE, **change})
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_predict_batch_rejects_empty_list():
    assert client.post("/predict_batch", json={"houses": []}).status_code == 422

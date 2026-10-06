import json

import pytest
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


def test_cors_lets_browser_apps_call_api():
    # Preflight a browser (e.g. a React app) sends before a cross-origin JSON POST
    r = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"


def test_predict_batch_rejects_empty_list():
    assert client.post("/predict_batch", json={"houses": []}).status_code == 422


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_predict_rejects_invalid_numbers(bad):
    # json.dumps writes NaN/Infinity tokens, which Python's JSON parser accepts
    body = json.dumps({**HOUSE, "population": bad})
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422

from app.models import price_model


def test_saved_model_meets_quality_bar():
    metrics = price_model.evaluate(price_model.get_model())
    assert metrics["mae"] < price_model.MAX_TEST_MAE
    assert metrics["r2"] > 0.8


def test_predictions_are_realistic_prices():
    # Tree models stay inside the training range; the linear model went negative
    tiny = {
        "longitude": -118.0, "latitude": 34.0, "housing_median_age": 0,
        "total_rooms": 1, "total_bedrooms": 0, "population": 0,
        "households": 1, "median_income": 0, "ocean_proximity": "INLAND",
    }
    assert 0 < price_model.predict([tiny])[0] < 550_000

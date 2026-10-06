from streamlit.testing.v1 import AppTest

from app.models import price_model

# Form defaults: first row of the dataset
DEFAULT_HOUSE = {
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


def run_app():
    # 60 s: the first run may need to download data and train the model
    return AppTest.from_file("../streamlit_app.py", default_timeout=60).run()


def test_form_shows_same_price_as_model():
    at = run_app()
    at.button[0].click().run()
    expected = price_model.predict([DEFAULT_HOUSE])[0]
    assert at.metric[0].value == f"${expected:,.0f}"


def test_form_shows_validation_errors():
    at = run_app()
    at.number_input(key="households").set_value(0)
    at.button[0].click().run()
    assert not at.metric
    assert "households" in at.error[0].value

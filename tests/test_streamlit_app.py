from streamlit.testing.v1 import AppTest

from app.models import price_model
from tests.test_api import HOUSE  # the form's defaults are the same first row


def test_form_shows_same_price_as_model():
    # 60 s: the first run may need to download data and train the model
    at = AppTest.from_file("../streamlit_app.py", default_timeout=60).run()
    at.button[0].click().run()
    expected = price_model.predict([HOUSE])[0]
    assert at.metric[0].value == f"${expected:,.0f}"

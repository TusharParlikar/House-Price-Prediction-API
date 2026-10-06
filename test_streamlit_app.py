from streamlit.testing.v1 import AppTest


def test_form_predicts_price_for_default_inputs():
    # 60 s: the first run may need to download data and train the model
    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    at.button[0].click().run()
    assert at.metric[0].value == "$428,125"

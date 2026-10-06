"""View: Streamlit web form on the same model the API uses.

It calls the model layer directly, so it works without the API running:
    streamlit run streamlit_app.py
"""
from typing import get_args

import streamlit as st
from pydantic import ValidationError

from app.models import price_model
from app.models.schemas import HouseData, OceanProximity

st.set_page_config(page_title="House Price Prediction")
st.title("House Price Prediction")
st.write(
    "Predicts the median house value of a California census block group "
    "(a small neighborhood) from 1990 census data. The defaults are the first "
    "row of the dataset, whose actual value is $452,600."
)

with st.form("house"):
    left, right = st.columns(2)
    house = {
        "longitude": left.number_input("Longitude", value=-122.23, key="longitude"),
        "latitude": right.number_input("Latitude", value=37.88, key="latitude"),
        "housing_median_age": left.number_input(
            "Median age of the houses (years)", value=41.0, key="housing_median_age"
        ),
        "total_rooms": right.number_input(
            "Total rooms in the block group", value=880.0, key="total_rooms"
        ),
        "total_bedrooms": left.number_input(
            "Total bedrooms in the block group", value=129.0, key="total_bedrooms"
        ),
        "population": right.number_input("Population", value=322.0, key="population"),
        "households": left.number_input("Households", value=126.0, key="households"),
        "median_income": right.number_input(
            "Median income (tens of thousands of USD, 8.3 = $83,000)",
            value=8.3252, format="%.4f", key="median_income",
        ),
        "ocean_proximity": st.selectbox(
            "Ocean proximity", get_args(OceanProximity), index=3, key="ocean_proximity"
        ),
    }
    submitted = st.form_submit_button("Predict price")

if submitted:
    try:
        HouseData(**house)  # same validation rules as the API
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(map(str, error["loc"])) or "input"
            st.error(f"{field}: {error['msg']}")
    else:
        price = price_model.predict([house])[0]
        st.metric("Predicted median house value", f"${price:,.0f}")

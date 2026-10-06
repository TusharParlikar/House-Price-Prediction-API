"""Web form for the house price model.

Loads the same model as the API, so it works without the API running:
    streamlit run streamlit_app.py
"""
import pandas as pd
import streamlit as st

from app.models.price_model import FEATURES, load_model

st.set_page_config(page_title="House Price Prediction")
st.title("House Price Prediction")
st.write(
    "Predicts the median house value of a California census block group "
    "(a small neighborhood) from 1990 census data. The defaults are the first "
    "row of the dataset, whose actual value is $452,600."
)

model = st.cache_resource(load_model)()

with st.form("house"):
    house = {
        "housing_median_age": st.number_input("Median age of the houses (years)", min_value=0.0, value=41.0),
        "total_rooms": st.number_input("Total rooms in the block group", min_value=0.0, value=880.0),
        "total_bedrooms": st.number_input("Total bedrooms in the block group", min_value=0.0, value=129.0),
        "population": st.number_input("Population", min_value=0.0, value=322.0),
        "households": st.number_input("Households", min_value=0.0, value=126.0),
        "median_income": st.number_input(
            "Median income (tens of thousands of USD, 8.3 = $83,000)",
            min_value=0.0, value=8.3252, format="%.4f",
        ),
    }
    submitted = st.form_submit_button("Predict price")

if submitted:
    price = model.predict(pd.DataFrame([house], columns=FEATURES))[0]
    st.metric("Predicted median house value", f"${price:,.0f}")

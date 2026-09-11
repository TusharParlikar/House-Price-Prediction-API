import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

FEATURES = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# --- Load the trained model ---
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="House Price Prediction API")


# --- Request schemas ---
class HouseData(BaseModel):
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float  # in tens of thousands, e.g. 8.3 = $83,000


class MultipleHouses(BaseModel):
    houses: List[HouseData]


# --- Routes ---
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}


@app.post("/predict")
def predict_price(data: HouseData):
    features = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)
    prediction = model.predict(features)
    return {"predicted_price": float(prediction[0])}


@app.post("/predict_batch")
def predict_batch(data: MultipleHouses):
    features = pd.DataFrame(
        [[getattr(house, f) for f in FEATURES] for house in data.houses],
        columns=FEATURES,
    )
    
    predictions = model.predict(features)
    return {"predicted_prices": predictions.tolist()}
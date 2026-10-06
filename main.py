from typing import Annotated, List

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from train import FEATURES, load_model

# --- Load the trained model ---
model = load_model()

app = FastAPI(title="House Price Prediction API")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # The default 422 echoes the bad input back, and NaN/Infinity can't be
    # written as JSON, so it would turn into a 500. Drop "input" instead.
    errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# --- Request schemas ---
# Counts, ages and income can't be negative; NaN/Infinity would crash the model
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class HouseData(BaseModel):
    housing_median_age: NonNegative
    total_rooms: NonNegative
    total_bedrooms: NonNegative
    population: NonNegative
    households: NonNegative
    median_income: NonNegative  # in tens of thousands, e.g. 8.3 = $83,000


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
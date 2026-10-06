from typing import Annotated

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
    # Sample request shown in /docs: first row of the dataset (actual value $452,600)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "housing_median_age": 41,
                    "total_rooms": 880,
                    "total_bedrooms": 129,
                    "population": 322,
                    "households": 126,
                    "median_income": 8.3252,
                }
            ]
        }
    }

    housing_median_age: NonNegative
    total_rooms: NonNegative
    total_bedrooms: NonNegative
    population: NonNegative
    households: NonNegative
    median_income: NonNegative  # in tens of thousands, e.g. 8.3 = $83,000


class MultipleHouses(BaseModel):
    houses: list[HouseData] = Field(min_length=1)


# --- Routes ---
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}


def run_model(houses):
    features = pd.DataFrame([h.model_dump() for h in houses], columns=FEATURES)
    return model.predict(features).tolist()


@app.post("/predict")
def predict_price(data: HouseData):
    return {"predicted_price": run_model([data])[0]}


@app.post("/predict_batch")
def predict_batch(data: MultipleHouses):
    return {"predicted_prices": run_model(data.houses)}

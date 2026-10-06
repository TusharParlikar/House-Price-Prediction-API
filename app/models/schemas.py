"""Model layer: request and response shapes, validated by Pydantic."""
from typing import Annotated

from pydantic import BaseModel, Field


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


# --- Response schemas ---
class Prediction(BaseModel):
    predicted_price: float  # median house value in USD


class BatchPrediction(BaseModel):
    predicted_prices: list[float]

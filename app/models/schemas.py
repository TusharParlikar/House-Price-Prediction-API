"""Model layer: request and response shapes, validated by Pydantic."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# --- Request schemas ---
# NaN/Infinity would crash the model, so every number must be finite
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
# The model divides by these (rooms per household, bedrooms per room)
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(allow_inf_nan=False)]
OceanProximity = Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class HouseData(BaseModel):
    # Sample request shown in /docs: first row of the dataset (actual value $452,600)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }

    longitude: Coordinate
    latitude: Coordinate
    housing_median_age: NonNegative
    total_rooms: Positive
    total_bedrooms: NonNegative
    population: NonNegative
    households: Positive
    median_income: NonNegative  # in tens of thousands, e.g. 8.3 = $83,000
    ocean_proximity: OceanProximity


class MultipleHouses(BaseModel):
    houses: list[HouseData] = Field(min_length=1)


# --- Response schemas ---
class Prediction(BaseModel):
    predicted_price: float  # median house value in USD


class BatchPrediction(BaseModel):
    predicted_prices: list[float]

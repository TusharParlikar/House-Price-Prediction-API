"""Controller layer: HTTP endpoints.

Each endpoint takes a validated request (app/models/schemas.py), asks the
model layer (app/models/price_model.py) for predictions and returns them.
"""
from fastapi import APIRouter

from app.models import price_model
from app.models.schemas import BatchPrediction, HouseData, MultipleHouses, Prediction

router = APIRouter()


# Health check: lets hosts (and people) see the API is up
@router.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}


# One block group in, one price out. FastAPI has already validated the body
# against HouseData, so bad input never reaches the model (it gets a 422).
@router.post("/predict", response_model=Prediction)
def predict_price(data: HouseData):
    return {"predicted_price": price_model.predict([data.model_dump()])[0]}


# Many block groups in one call; one model call for all of them
@router.post("/predict_batch", response_model=BatchPrediction)
def predict_batch(data: MultipleHouses):
    return {"predicted_prices": price_model.predict([h.model_dump() for h in data.houses])}

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.price_model import FEATURES, load_model
from app.models.schemas import BatchPrediction, HouseData, MultipleHouses, Prediction

# --- Load the trained model ---
model = load_model()

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts the median house value (USD) of a California census block group.",
)

# Public API: let browser apps (React etc.) on any site call it. No cookies
# or auth are involved, so allowing every origin exposes nothing extra.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # The default 422 echoes the bad input back, and NaN/Infinity can't be
    # written as JSON, so it would turn into a 500. Drop "input" instead.
    errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# --- Routes ---
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}


def run_model(houses):
    features = pd.DataFrame([h.model_dump() for h in houses], columns=FEATURES)
    return model.predict(features).tolist()


@app.post("/predict", response_model=Prediction)
def predict_price(data: HouseData):
    return {"predicted_price": run_model([data])[0]}


@app.post("/predict_batch", response_model=BatchPrediction)
def predict_batch(data: MultipleHouses):
    return {"predicted_prices": run_model(data.houses)}

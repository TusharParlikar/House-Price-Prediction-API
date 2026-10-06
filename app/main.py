"""FastAPI application: connects the controllers to the web server.

    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.controllers import prediction_controller
from app.models import price_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    price_model.get_model()  # load (or train) the model before the first request
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predicts the median house value (USD) of a California census block group.",
    lifespan=lifespan,
)

# Public API: let browser apps (React etc.) on any site call it. No cookies
# or auth are involved, so allowing every origin exposes nothing extra.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # The default 422 echoes the bad input back, and NaN/Infinity can't be
    # written as JSON, so it would turn into a 500. Keep type, loc and msg only
    # ("ctx" repeats msg, and holds an unprintable exception for custom rules).
    errors = [{k: e[k] for k in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


app.include_router(prediction_controller.router)

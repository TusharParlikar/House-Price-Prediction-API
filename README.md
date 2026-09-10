# House Price Prediction API

A machine learning API that predicts California housing prices. A Linear
Regression model (scikit-learn) is trained on the **real California
Housing dataset** — 1990 U.S. census data covering ~20,640 block groups
— exported as a pickle file, and served through a FastAPI backend with
single and batch prediction endpoints.

## About the Data

This uses the well-known California Housing dataset (Pace & Barry, 1997),
loaded directly from a public GitHub source at train time — no manual
download needed. Each row is a **census block group**, not an individual
house, with features like housing age, room/bedroom counts, population,
household count, and median income, against the block group's median
house value.

## What's Included

- A training step that pulls the real dataset, splits it into train/test
  sets, trains the model, and prints accuracy metrics (R² and mean
  absolute error).
- A REST API with a welcome endpoint, a single-prediction endpoint, and
  a batch prediction endpoint for multiple entries at once.
- A small script for testing the live API once it's running.
- Auto-generated interactive API documentation (Swagger UI).

## How to Use This Project

1. **Get the code** — pull all files from `project.md` into a folder, or
   clone the repo if you've already pushed it to GitHub.
2. **Install dependencies** — set up a virtual environment and install
   everything listed in `requirements.txt`.
3. **Train the model** — run the training script once. It downloads the
   dataset, trains the model, prints accuracy metrics, and saves the
   trained model file. Re-run this any time you want to retrain.
4. **Start the API** — launch the server. It runs locally and reloads
   automatically as you make changes.
5. **Try it out** — visit the interactive docs in your browser to test
   endpoints directly, run the included test script, or send requests
   with a tool like curl or Postman.

## Endpoints

| Endpoint         | Purpose                                          |
|-------------------|---------------------------------------------------|
| Welcome route     | Confirms the API is running                       |
| Single prediction | Predicts median house value for one block group   |
| Batch prediction  | Predicts values for a list of entries at once      |

## Model Inputs

Each prediction needs: housing median age, total rooms, total bedrooms,
population, households, and median income (in tens of thousands — e.g.
5.2 means $52,000).

## Notes

- This is real, publicly available data, but it's aggregated per census
  block group rather than per individual home sale — treat predictions
  as neighborhood-level estimates, not per-house appraisals.
- Plain Linear Regression gets an R² around 0.57 on this data — a
  reasonable baseline. Feature engineering (e.g. rooms-per-household) or
  a different model (Random Forest, Gradient Boosting) would improve it.
- The trained model file isn't tracked in version control — anyone using
  this project regenerates it locally by running the training step.
- For real deployment, run the server without auto-reload and with
  multiple worker processes instead.

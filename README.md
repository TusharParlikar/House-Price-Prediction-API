# House Price Prediction API

A FastAPI service that predicts median house values for California neighborhoods, built on the public California Housing dataset.

You send six numbers that describe a census block group (house age, rooms, bedrooms, population, households, median income). The API returns the predicted median house value in US dollars. A scikit-learn Linear Regression model does the prediction. The training script downloads the data and builds the model, so the repository contains no data or model files.

## Features

- **Single prediction**: `POST /predict` returns a price for one block group.
- **Batch prediction**: `POST /predict_batch` returns prices for a list of block groups in one call.
- **Input validation**: negative, missing, NaN or Infinity values get a `422` error that names the bad field.
- **Interactive docs**: Swagger UI at `/docs` with a ready-to-run sample request.
- **One-command training**: `python train.py` downloads the data, trains, prints accuracy and saves the model.
- **No manual setup for the model**: if the model file is missing, the API trains one on first start.

## Quick start

Prerequisites: Python 3.11 or newer (tested on 3.14) and internet access for the first training run.

```bash
git clone https://github.com/TusharParlikar/House-Price-Prediction-API.git
cd House-Price-Prediction-API
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python train.py                # optional: the API trains on first start if needed
uvicorn main:app --reload
```

`python train.py` prints:

```text
R^2 score:  0.571
MAE:        $56,642
Model saved to .../house_price_model.pkl
```

Open http://127.0.0.1:8000/docs, choose `POST /predict`, then **Try it out** and **Execute**.

## API

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/` | none | `{"message": "Welcome to the House Price Prediction API"}` |
| `POST` | `/predict` | one house object | `{"predicted_price": 428125.14}` |
| `POST` | `/predict_batch` | `{"houses": [house, ...]}` (at least 1) | `{"predicted_prices": [428125.14, ...]}` |

FastAPI also serves `/docs` (Swagger UI), `/redoc` and `/openapi.json`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129,
       "population": 322, "households": 126, "median_income": 8.3252}'
```

```json
{"predicted_price": 428125.14360750734}
```

This is the first row of the dataset. Its real median value is $452,600.

From Python:

```python
import requests

house = {"housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129,
         "population": 322, "households": 126, "median_income": 8.3252}
r = requests.post("http://127.0.0.1:8000/predict_batch", json={"houses": [house, house]})
print(r.json())  # {'predicted_prices': [428125.14..., 428125.14...]}
```

### Input fields

All six fields are required, and each must be a finite number ≥ 0. Each request describes one census **block group** (a small neighborhood), not one house.

| Field | Meaning | Range in the training data |
|-------|---------|----------------------------|
| `housing_median_age` | Median age of the houses, in years | 1 to 52 |
| `total_rooms` | Total rooms in the whole block group | 2 to 39,320 |
| `total_bedrooms` | Total bedrooms in the whole block group | 1 to 6,445 |
| `population` | People living in the block group | 3 to 35,682 |
| `households` | Households in the block group | 1 to 6,082 |
| `median_income` | Median household income in tens of thousands of USD (`8.3` = $83,000) | 0.5 to 15.0 |

### Errors

Invalid input returns `422` with one entry per problem:

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "population"],
             "msg": "Input should be greater than or equal to 0", "ctx": {"ge": 0.0}}]}
```

## How it works

```mermaid
flowchart LR
    CSV["housing.csv<br/>(GitHub)"] -->|train.py| PKL["house_price_model.pkl"]
    PKL -->|load_model| API["main.py<br/>(FastAPI)"]
    Client -->|"POST /predict<br/>POST /predict_batch"| API
```

*Where the model comes from and how requests reach it.*

- **Data**: the California Housing dataset (Pace & Barry, 1997), 1990 US census, 20,640 block groups. `train.py` reads it from the [handson-ml2 repository](https://github.com/ageron/handson-ml2/tree/master/datasets/housing).
- **Cleaning**: the 207 rows with no `total_bedrooms` are dropped, which leaves 20,433 rows.
- **Model**: `LinearRegression` on the six input fields, with an 80/20 train/test split (`random_state=42`).
- **Accuracy on the test set**: R² 0.571, mean absolute error $56,642.
- **Serving**: `main.py` loads the model once at startup through `load_model()` in `train.py`. If `house_price_model.pkl` doesn't exist, `load_model()` trains a new one first.

`House_rate_prediction_API.ipynb` runs the same training steps one at a time, with output after each step.

## Project structure

```text
main.py                          FastAPI app: request schemas, validation, endpoints
train.py                         Downloads data, trains, evaluates, saves the model
House_rate_prediction_API.ipynb  The same training steps as a notebook
test_main.py                     API tests (pytest + FastAPI TestClient)
requirements.txt                 Pinned dependencies
house_price_model.pkl            Trained model, created by train.py (gitignored)
```

## Tests

```bash
python -m pytest
```

The tests check every endpoint, check that single and batch predictions agree, and check the `422` errors for negative, NaN, Infinity and empty-batch input. If no model file exists yet, the first run trains one, which needs internet access.

## Running in production

Train once, then start the server without `--reload`:

```bash
python train.py
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Train before starting several workers. Otherwise, each worker trains its own model at the same time and writes the same file.

## Limitations

- **Baseline accuracy**: predictions are off by $56,642 on average (R² 0.571). Adding features (location, rooms per household) or using a tree-based model would improve this.
- **Location is ignored**: the dataset has `longitude`, `latitude` and `ocean_proximity`, but the model doesn't use them. Location affects price a lot.
- **Old, capped data**: values are from the 1990 census, in 1990 dollars. The dataset caps house values at $500,001, so the model has no examples above that.
- **Linear model**: inputs far outside the training ranges give unrealistic results. For example, all zeros predicts −$45,966.
- **Neighborhood level only**: a prediction is a block-group median, not an appraisal of one house.
- **No authentication or rate limiting**: anyone who can reach the server can call it.
- **Pickle file**: `pickle.load` can run code from the file it loads. Only load a model you trained yourself.

## License

No license file yet.

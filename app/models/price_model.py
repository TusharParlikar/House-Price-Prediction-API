"""Model layer: train, save, load and run the house price model.

The model is one scikit-learn pipeline:
    add ratio features -> one-hot encode ocean_proximity -> gradient boosting
so callers pass the 9 raw input fields and get a price in USD back.
"""
import functools
import pickle
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "housing.csv"  # local copy of DATA_URL
MODEL_PATH = ROOT / "artifacts" / "house_price_model.pkl"

FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]
TARGET = "median_house_value"
# Quality bar: train() refuses to save a model whose test error is above this.
# Current model: ~$29,400. Old linear model: $56,642.
MAX_TEST_MAE = 35_000


def load_data():
    """Return the dataset, downloading it to data/ on first use."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    # total_bedrooms has 207 missing values out of 20,640 rows - drop them
    return pd.read_csv(DATA_PATH).dropna(subset=FEATURES)


def split_data():
    """Same 80/20 train/test split every time (random_state=42)."""
    data = load_data()
    return train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=42)


def add_ratio_features(X):
    """Per-household ratios describe a neighborhood better than raw totals."""
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"]
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
    X["people_per_household"] = X["population"] / X["households"]
    return X


def build_pipeline():
    return make_pipeline(
        FunctionTransformer(add_ratio_features),
        ColumnTransformer(
            [("ocean", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ocean_proximity"])],
            remainder="passthrough",
        ),
        HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, random_state=42
        ),
    )


def evaluate(model):
    """R^2 and mean absolute error (USD) on the held-out test set."""
    _, X_test, _, y_test = split_data()
    preds = model.predict(X_test)
    return {"r2": r2_score(y_test, preds), "mae": mean_absolute_error(y_test, preds)}


def train():
    X_train, _, y_train, _ = split_data()

    # 5-fold cross-validation on the training set: is the error stable?
    cv_mae = -cross_val_score(
        build_pipeline(), X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"Cross-validation MAE: ${cv_mae.mean():,.0f} (+/- ${cv_mae.std():,.0f})")

    model = build_pipeline().fit(X_train, y_train)
    metrics = evaluate(model)
    print(f"Test R^2: {metrics['r2']:.3f}")
    print(f"Test MAE: ${metrics['mae']:,.0f}")
    if metrics["mae"] > MAX_TEST_MAE:
        raise RuntimeError(
            f"Test MAE ${metrics['mae']:,.0f} is above the ${MAX_TEST_MAE:,} bar; "
            "model not saved. Check the data source and training settings."
        )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    return model


@functools.cache
def get_model():
    """Load the saved model once per process, training it first if the file is missing."""
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH.name} not found, training a new model...")
        return train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(houses):
    """Return the predicted median house value (USD) for each house dict."""
    features = pd.DataFrame(houses, columns=FEATURES)
    return get_model().predict(features).tolist()

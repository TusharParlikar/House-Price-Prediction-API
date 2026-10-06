"""Train the house price model and save it to house_price_model.pkl.

Same steps as House_rate_prediction_API.ipynb, runnable without Jupyter:
    python train.py
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
MODEL_PATH = Path(__file__).with_name("house_price_model.pkl")

FEATURES = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
TARGET = "median_house_value"


def train():
    # --- Load real data ---
    data = pd.read_csv(DATA_URL)
    # total_bedrooms has ~207 missing values out of 20,640 rows - drop them
    data = data.dropna(subset=FEATURES)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)

    # --- Evaluate on held-out data ---
    preds = model.predict(X_test)
    print(f"R^2 score:  {r2_score(y_test, preds):.3f}")
    print(f"MAE:        ${mean_absolute_error(y_test, preds):,.0f}")

    # --- Export model ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model():
    """Load the saved model, training a new one first if the file is missing."""
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH.name} not found, training a new model...")
        return train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train()

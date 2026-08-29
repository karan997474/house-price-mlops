from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "name": name,
        "model_object": model,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2_score": float(r2_score(y_test, predictions)),
    }


def train_model():
    print("Loading training data...")

    train_data = pd.read_csv(ARTIFACTS_DIR / "train.csv")
    test_data = pd.read_csv(ARTIFACTS_DIR / "test.csv")

    X_train = train_data.drop("price", axis=1)
    y_train = train_data["price"]

    X_test = test_data.drop("price", axis=1)
    y_test = test_data["price"]

    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    results = []

    results.append(
        evaluate_model(
            "LinearRegression",
            linear_model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "RandomForestRegressor",
            random_forest,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    best_result = max(results, key=lambda result: result["r2_score"])

    joblib.dump(best_result["model_object"], MODEL_PATH)

    metrics = {
        "best_model": best_result["name"],
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "features": list(X_train.columns),
        "models": {},
    }

    for result in results:
        metrics["models"][result["name"]] = {
            "mae": result["mae"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "r2_score": result["r2_score"],
        }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n===== MODEL COMPARISON =====")

    for result in results:
        print(f"\n{result['name']}")
        print(f"MAE:  {result['mae']:.2f}")
        print(f"RMSE: {result['rmse']:.2f}")
        print(f"R²:   {result['r2_score']:.4f}")

    print("\n============================")
    print(f"BEST MODEL: {best_result['name']}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()

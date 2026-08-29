from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="House Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

model = joblib.load(MODEL_PATH)


class HouseData(BaseModel):
    area: float = Field(gt=0, le=100000)
    bedrooms: int = Field(ge=1, le=20)
    age: float = Field(ge=0, le=200)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(data: HouseData):
    input_data = pd.DataFrame(
        [{
            "area": data.area,
            "bedrooms": data.bedrooms,
            "age": data.age,
        }]
    )

    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}

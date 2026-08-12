from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("artifacts/model.pkl")


class HouseData(BaseModel):
    area: float
    bedrooms: int
    age: float


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseData):
    input_data = pd.DataFrame([{
        "area": data.area,
        "bedrooms": data.bedrooms,
        "age": data.age
    }])

    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}

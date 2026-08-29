# House Price MLOps

An end-to-end machine learning project that predicts house prices from area, bedrooms, and property age. The trained pipeline is served through a FastAPI API and can run locally or in Docker.

## Features

- Data ingestion and train/test split
- Scikit-learn pipeline with feature scaling and Linear Regression
- Saved trained model pipeline
- FastAPI prediction API
- Input validation for house details
- Health-check endpoint
- Docker support

## Project Structure

```text
house-price-mlops/
├── app.py
├── data.csv
├── requirements.txt
├── Dockerfile
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   └── model.pkl
└── src/components/
    ├── data_ingestion.py
    ├── data_transformation.py
    └── model_trainer.py

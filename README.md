# House Price MLOps

A simple end-to-end Machine Learning project that predicts house prices using a trained ML model and serves predictions through a FastAPI API.

## Features

* House price prediction using Machine Learning
* Data ingestion and model training pipeline
* Trained model saved as `model.pkl`
* FastAPI API for predictions
* Docker support
* Git and GitHub for version control

## Project Structure

```text
house-price-mlops/
│
├── app.py
├── data.csv
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   └── model.pkl
│
└── src/
    └── components/
        ├── data_ingestion.py
        ├── data_transformation.py
        └── model_trainer.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/karan997474/house-price-mlops.git
```

Go inside the project folder:

```bash
cd house-price-mlops
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

## How It Works

The basic flow of the project is:

```text
House Data
    ↓
Data Ingestion
    ↓
Train / Test Split
    ↓
Model Training
    ↓
model.pkl
    ↓
FastAPI
    ↓
House Price Prediction
```

## Future Improvements

* Add Docker container testing
* Add automated testing
* Add GitHub Actions CI/CD
* Deploy the API to cloud
* Add model monitoring
* Add model versioning

## Author

Karan
GitHub: `karan997474`


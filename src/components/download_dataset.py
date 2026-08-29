from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data.csv"


def download_dataset():
    print("Downloading Ames Housing dataset...")

    housing = fetch_openml(
        name="house_prices",
        as_frame=True,
        parser="auto"
    )

    df = housing.frame.copy()

    # Keep useful features for our simple API
    df = df[
        [
            "GrLivArea",
            "BedroomAbvGr",
            "YearBuilt",
            "YrSold",
            "SalePrice",
        ]
    ].copy()

    # Convert original columns into our API format
    df["age"] = df["YrSold"] - df["YearBuilt"]

    df = df.rename(
        columns={
            "GrLivArea": "area",
            "BedroomAbvGr": "bedrooms",
            "SalePrice": "price",
        }
    )

    df = df[
        [
            "area",
            "bedrooms",
            "age",
            "price",
        ]
    ]

    # Clean invalid/missing rows
    df = df.dropna()

    df = df[
        (df["area"] > 0)
        & (df["bedrooms"] >= 1)
        & (df["age"] >= 0)
    ]

    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset prepared successfully.")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    download_dataset()

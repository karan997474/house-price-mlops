import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    data = pd.read_csv("data.csv")

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    os.makedirs("artifacts", exist_ok=True)

    train_data.to_csv("artifacts/train.csv", index=False)
    test_data.to_csv("artifacts/test.csv", index=False)

    print("Data ingestion completed successfully.")


if __name__ == "__main__":
    load_data()

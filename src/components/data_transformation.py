import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform_data():
    train_data = pd.read_csv("artifacts/train.csv")
    test_data = pd.read_csv("artifacts/test.csv")

    X_train = train_data.drop("price", axis=1)
    y_train = train_data["price"]

    X_test = test_data.drop("price", axis=1)
    y_test = test_data["price"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data transformation completed successfully.")


if __name__ == "__main__":
    transform_data()

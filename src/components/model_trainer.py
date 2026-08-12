import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model():
    train_data = pd.read_csv("artifacts/train.csv")
    test_data = pd.read_csv("artifacts/test.csv")

    X_train = train_data.drop("price", axis=1)
    y_train = train_data["price"]

    X_test = test_data.drop("price", axis=1)
    y_test = test_data["price"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "artifacts/model.pkl")

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print("Model training completed successfully.")
    print("MSE:", mse)


if __name__ == "__main__":
    train_model()

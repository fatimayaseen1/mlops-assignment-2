import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression


def load_data(path):
    return pd.read_csv(path)


def train_model(data):
    X = data[["feature"]]
    y = data["label"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def save_model(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    data = load_data("data/dataset.csv")
    model = train_model(data)
    save_model(model, "models/model.pkl")
    print("Model trained and saved successfully.")


# src/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model():
    data = pd.read_csv("data/dataset.csv")
    print(data.head()) 
    X = data[["feature"]]
    y = data["label"]

    model = LinearRegression()
    model.fit(X, y)

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import os

DATA_PATH = "data/dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/model.pkl"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    X = data[["feature"]]
    y = data["label"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    main()
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model trained and saved successfully.")
    return model

if __name__ == "__main__":
    train_model()

import os
import pandas as pd
from src.train import load_data, train_model, save_model


def test_train_model_runs():
    data = load_data("data/dataset.csv")
    model = train_model(data)
    assert model is not None


def test_model_file_created():
    data = load_data("data/dataset.csv")
    model = train_model(data)
    save_model(model, "models/model.pkl")
    assert os.path.exists("models/model.pkl")


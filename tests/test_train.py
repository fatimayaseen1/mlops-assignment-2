import pandas as pd
import os


def test_data_loading():
    data = pd.read_csv("data/dataset.csv")
    assert not data.empty


def test_columns_exist():
    data = pd.read_csv("data/dataset.csv")
    assert "feature" in data.columns
    assert "label" in data.columns


def test_training_output():
    os.system("python src/train.py")
    assert os.path.exists("models/model.pkl")

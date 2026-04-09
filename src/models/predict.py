import pandas as pd

import joblib

from src.data.load import load_raw_data
from src.data.preprocess import clean_data

def main():
    # Load model
    model = joblib.load("models/best_model.pkl")

    # Load new data
    df = load_raw_data("test.csv")
    df = clean_data(df)

    # Predict
    predictions = model.predict(df)

    print(predictions[:10])
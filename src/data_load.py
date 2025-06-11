# data_loader.py
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
FEATURE = "Internal Unit Sales Projections"  # Remove trailing space
TARGET = "Gross Sales ($)"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(df.head())  # Debug: Check loaded data

    df.columns = df.columns.str.strip()  # ✅ Strip spaces from column names

    # Clean numeric columns
    df[FEATURE] = df[FEATURE].astype(str).str.replace(",", "").astype(float)
    df[TARGET] = df[TARGET].astype(str).str.replace(",", "").astype(float)

    df = df.dropna(subset=[FEATURE, TARGET])
    X = df[[FEATURE]].values
    y = df[TARGET].values
    return X, y

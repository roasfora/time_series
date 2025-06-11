import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ---- Config ---- #
FEATURE = "Internal Unit Sales Projections"
TARGET = "Gross Sales ($)"
FORECAST_OUTPUT_DIR = os.getcwd()  # Save in the same folder as the script

# ---- Load Data ---- #
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Clean and convert columns
df[FEATURE] = df[FEATURE].astype(str).str.replace(",", "").astype(float)
df[TARGET] = df[TARGET].astype(str).str.replace(",", "").astype(float)

# ---- Create Lag Features ---- #
df["lag_1"] = df[FEATURE].shift(1)
df["lag_2"] = df[FEATURE].shift(2)

# Filter for training: drop rows with NaNs in lags or target
df_train = df.dropna(subset=[FEATURE, TARGET, "lag_1", "lag_2"])
X = df_train[[FEATURE, "lag_1", "lag_2"]].values
y = df_train[TARGET].values

# ---- Train/Test Split ---- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Prepare Future Inputs ---- #
df_future = df[df[TARGET].isna()].copy()
df_future["lag_1"] = df[FEATURE].shift(1)
df_future["lag_2"] = df[FEATURE].shift(2)
df_future = df_future.dropna(subset=["lag_1", "lag_2"])
X_future = df_future[[FEATURE, "lag_1", "lag_2"]].values

# ---- Define Models ---- #
models = {
    "linear": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "xgboost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

# ---- Train, Evaluate, Forecast ---- #
metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics.append({"Model": name, "MAPE": round(mape, 4), "R2": round(r2, 4)})

    print(f"\n{name.upper()}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²:   {r2:.4f}")

    # Forecast
    if X_future.shape[0] > 0:
        df_future[name] = model.predict(X_future)
        df_future.to_csv(os.path.join(FORECAST_OUTPUT_DIR, f"{name}_forecast.csv"), index=False)

    # Plot actual vs predicted
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name.upper()} Forecast\nMAPE: {mape:.2%}, R²: {r2:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FORECAST_OUTPUT_DIR, f"{name}_forecast.png"))
    plt.close()

# ---- Save Metrics ---- #
pd.DataFrame(metrics).to_csv(os.path.join(FORECAST_OUTPUT_DIR, "model_metrics.csv"), index=False)

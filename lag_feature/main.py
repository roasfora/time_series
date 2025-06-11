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

df[FEATURE] = df[FEATURE].astype(str).str.replace(",", "").astype(float)
df[TARGET] = df[TARGET].astype(str).str.replace(",", "").astype(float)

df_train = df.dropna(subset=[FEATURE, TARGET])
X = df_train[[FEATURE]].values
y = df_train[TARGET].values

# ---- Train/Test Split ---- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Prepare Future Inputs ---- #
df_future = df[df[TARGET].isna()].copy()
X_future = df_future[[FEATURE]].values

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

        # Save forecast
        forecast_path = os.path.join(FORECAST_OUTPUT_DIR, f"{name}_forecast.csv")
        df_future.to_csv(forecast_path, index=False)

    # Plot actual vs predicted
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name.upper()} Forecast\nMAPE: {mape:.2%}, R²: {r2:.4f}")
    plt.tight_layout()

    plot_path = os.path.join(FORECAST_OUTPUT_DIR, f"{name}_forecast.png")
    plt.savefig(plot_path)
    plt.close()

# ---- Save Metrics ---- #
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(os.path.join(FORECAST_OUTPUT_DIR, "model_metrics.csv"), index=False)

# main.py
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data_load import load_data
from train import get_models
from evaluate import evaluate
from plot import save_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load historical training data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load future input data
    future_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
    future_df = pd.read_csv(future_path)
    future_df.columns = future_df.columns.str.strip()

    # Prepare future input features
    input_col = "Internal Unit Sales Projections"
    future_df[input_col] = future_df[input_col].astype(str).str.replace(",", "").astype(float)
    X_future = future_df[[input_col]].values

    # Get models
    models = get_models()
    logger.info(f"Training {len(models)} models...")

    # Collect metrics
    metrics = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        mape, r2 = evaluate(y_test, y_pred)

        logger.info(f"Model: {name}")
        logger.info(f"  MAPE: {mape:.4f}")
        logger.info(f"  R²: {r2:.4f}")

        # Store metrics
        metrics.append({
            "Model": name,
            "MAPE": round(mape, 4),
            "R2": round(r2, 4)
        })

        # Forecast future
        future_predictions = model.predict(X_future)
        future_df[name] = future_predictions

        # Save plot and forecast
        save_plot(y_test, y_pred, name, mape, r2, forecast_df=future_df.copy())

    # Save all model metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)

if __name__ == "__main__":
    main()

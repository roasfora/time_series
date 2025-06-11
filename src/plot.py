import matplotlib.pyplot as plt
import os
import pandas as pd

def save_plot(y_true, y_pred, model_name, mape, r2, forecast_df=None):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Forecast\nMAPE: {mape:.2%}, R²: {r2:.4f}")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'{model_name}_forecast.png')
    plt.savefig(plot_path)
    plt.close()

    # Save forecast CSV if provided
    if forecast_df is not None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'{model_name}_forecast.csv')
        forecast_df.to_csv(csv_path, index=False)

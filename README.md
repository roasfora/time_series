
# Levi Forecasting Project

## 📈 Overview
This project forecasts Levi Strauss's gross sales based on internal sales projections using machine learning models. The goal is to replicate and enhance the AI-based forecasting approach used by Levi in collaboration with Wipro, evaluating performance through MAPE and R².

Recent improvements to the model include the incorporation of lag features, which have further enhanced performance by providing temporal context to the forecasts.

---

## 🔧 Project Structure

```
levi_project/
├── data/                  # Input CSVs and output predictions/plots
│   ├── data.csv
│   ├── model_metrics.csv
│   └── *_forecast.csv/png
├── main.py                # Single-run pipeline script
├── train.py               # Model definitions
├── evaluate.py            # Evaluation metrics (MAPE, R²)
├── data_loader.py         # Load and clean input data
├── plot.py                # Save forecasts and visuals
└── README.md
```

---

## 📂 Data Format

The primary dataset is `data/data.csv`, which includes:

| Year | Month | Internal Unit Sales Projections | Gross Sales ($) |
|------|-------|---------------------------------|-----------------|
| 2022 | 1     | 7,500,000                       |  ...            |
| 2022 | 2     | 8,700,000                       |  ...            |
| ...  | ...   | ...                             |  ...            |

Note: Forecasting will be applied to rows where `Gross Sales ($)` is missing.

---

## ⚙️ How It Works

1. **Load & Clean Data** – Removes formatting issues, handles missing values.
2. **Split Train/Test** – 80/20 split using Scikit-learn.
3. **Train Models** – Linear Regression, Random Forest, XGBoost.
4. **Evaluate** – Report MAPE and R² scores.
5. **Forecast** – Predict future missing values in the dataset.
6. **Visualize** – Save actual vs predicted scatter plots.
7. **Save Outputs** – CSV files for forecasts and metrics.

---

## 📊 Results

| Model         | MAPE   | R²     |
|---------------|--------|--------|
| Linear        | 0.0387 | 0.9700 |
| Random Forest | 0.0569 | 0.9064 |
| XGBoost       | 0.0735 | 0.8580 |

✅ **Linear Regression** had the best accuracy with only one feature, suggesting a strong linear correlation.

---

## 📘 Enhancements: Lag Features

The updated model includes **lag features**, which incorporate the values from previous months. This adds valuable temporal structure and allows models like Random Forest and XGBoost to leverage sequential patterns.

➡️ With lag features:
- Accuracy improved
- Seasonality became easier to capture
- Linear regression still performed well, but tree models showed more promise

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/levi-forecasting.git
cd levi-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install required libraries
pip install -r requirements.txt
```

Required libraries:
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`

---

## ▶️ Run the Project

To run the full pipeline:

```bash
python main.py
```

Output files will be saved in the `data/` folder.

---

## 🚀 Future Ideas

- Add a Streamlit app for interactive forecasting (optimistic, regular, pessimistic).
- Integrate external data like CPI (inflation).
- Apply cross-validation and hyperparameter tuning.
- Transition to time series models like ARIMA, LSTM if temporal dynamics grow.

---

## 🧠 Conclusion

The combination of solid data cleaning, a simple feature set, and classical regression models can yield highly accurate forecasts — especially when enhanced with lagged variables. This supports business decision-making with speed, transparency, and effectiveness.

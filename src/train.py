# train.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_models():
    return {
        "linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "xgboost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }
# evaluate.py
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert len(y_true) == len(y_pred), "Length mismatch!"
    
    # Handle zeros in y_true
    if np.any(y_true == 0):
        print("Warning: y_true contains zeros. Adding epsilon=1e-6 to avoid division by zero.")
        y_true = y_true + 1e-6
    
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mape, r2
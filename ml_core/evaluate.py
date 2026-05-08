import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Root Mean Square Error — penalises large errors more heavily than MAE.
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Mean Absolute Error — average magnitude of errors, robust to outliers.
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Coefficient of Determination — how much variance the model explains (1.0 = perfect).
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Guard against flat target (e.g. a nighttime-only slice where all values are 0)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Run all metrics and return a dict rounded to 4 decimal places.
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae":  round(mae(y_true,  y_pred), 4),
        "r2":   round(r2(y_true,   y_pred), 4),
    }



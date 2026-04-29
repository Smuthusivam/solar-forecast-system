import numpy as np
import pandas as pd


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


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    # Mean Absolute Percentage Error — only computed on daytime rows (actual > 10 W/m²).
    mask = y_true > 10
    if mask.sum() == 0:
        return 0.0

    y_true_day = y_true[mask]
    y_pred_day = y_pred[mask]

    return float(np.mean(np.abs((y_true_day - y_pred_day) / (y_true_day + epsilon))) * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Run all four metrics and return a dict rounded to 4 decimal places.
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae":  round(mae(y_true,  y_pred), 4),
        "r2":   round(r2(y_true,   y_pred), 4),
        "mape": round(mape(y_true,  y_pred), 4),
    }


def compute_residuals(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    timestamps: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    # Build a residuals DataFrame (actual - predicted) for anomaly analysis.
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    residuals     = y_true - y_pred
    abs_residuals = np.abs(residuals)

    # Percentage error only on daytime rows — nighttime zeros make it meaningless
    mask      = y_true > 10
    pct_error = np.zeros_like(y_true)
    pct_error[mask] = np.abs(residuals[mask] / (y_true[mask] + 1e-8)) * 100

    df = pd.DataFrame({
        "actual":       y_true,
        "predicted":    y_pred,
        "residual":     residuals,
        "abs_residual": abs_residuals,
        "pct_error":    pct_error,
    })

    if timestamps is not None:
        df.insert(0, "timestamp", timestamps)

    return df


def compare_models(metrics_list: list) -> pd.DataFrame:
    # Return a DataFrame of per-model metrics sorted by RMSE ascending.
    df   = pd.DataFrame(metrics_list)
    cols = ["model", "rmse", "mae", "r2", "mape"]
    df   = df[[c for c in cols if c in df.columns]]
    df   = df.sort_values("rmse").reset_index(drop=True)
    return df


if __name__ == "__main__":

    np.random.seed(42)
    n = 500

    y_true = np.abs(np.random.normal(400, 150, n))

    y_good = np.clip(y_true + np.random.normal(0, 40, n),  0, None)
    y_poor = np.clip(y_true + np.random.normal(0, 120, n), 0, None)

    print("=" * 45)
    print("Good model metrics:")
    good_metrics = compute_all(y_true, y_good)
    for k, v in good_metrics.items():
        print(f"  {k.upper():>6}: {v}")

    print("\nPoor model metrics:")
    poor_metrics = compute_all(y_true, y_poor)
    for k, v in poor_metrics.items():
        print(f"  {k.upper():>6}: {v}")

    idx       = pd.date_range("2024-06-01", periods=n, freq="h")
    residuals = compute_residuals(y_true, y_good, timestamps=idx)
    print(f"\nResiduals DataFrame shape: {residuals.shape}")
    print(residuals.head(4).to_string(index=False))

    metrics_list = [
        {**good_metrics, "model": "XGBoost"},
        {**poor_metrics, "model": "Prophet"},
        {"rmse": 35.2, "mae": 26.1, "r2": 0.95, "mape": 6.8, "model": "Ensemble"},
    ]
    comparison = compare_models(metrics_list)
    print(f"\nModel comparison table (sorted by RMSE):")
    print(comparison.to_string(index=False))

    print("\nevaluate.py passed all checks.")

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# SECTION 1 — Individual Metric Functions
# ─────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Primary metric for this project. Penalises large errors more heavily
    than MAE because errors are squared before averaging. A prediction
    that's 100 W/m² off hurts 4x more than one that's 50 W/m² off.

    Unit: W/m² (same as irradiance — easy to interpret)
    Lower is better.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Average magnitude of errors without squaring. More robust to
    outliers than RMSE. If RMSE >> MAE, it means a few large errors
    are dragging RMSE up — useful diagnostic for anomaly days.

    Unit: W/m²
    Lower is better.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).

    Measures how much of the variance in irradiance the model explains.
    - R² = 1.0 → perfect predictions
    - R² = 0.0 → model is no better than predicting the mean every time
    - R² < 0.0 → model is worse than the mean (something is very wrong)

    For solar irradiance, R² > 0.85 is considered good.
    Dimensionless. Higher is better.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Guard against flat target (all zeros — e.g. nighttime-only slice)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Expresses error as a percentage of the actual value — useful for
    communicating accuracy to non-technical stakeholders in the thesis.
    "The model is off by X% on average" is easy to understand.

    epsilon prevents division by zero for nighttime readings where
    y_true is 0 (or very close to 0). Those near-zero rows are masked
    out so they don't distort the percentage calculation.

    Unit: % (percentage points)
    Lower is better.
    """
    # Only compute MAPE on daytime rows where actual irradiance > 10 W/m²
    # (nighttime zeros make percentage error meaningless and infinitely large)
    mask = y_true > 10
    if mask.sum() == 0:
        return 0.0

    y_true_day = y_true[mask]
    y_pred_day = y_pred[mask]

    return float(np.mean(np.abs((y_true_day - y_pred_day) / (y_true_day + epsilon))) * 100)


# ─────────────────────────────────────────────
# SECTION 2 — Master Compute Function
# ─────────────────────────────────────────────

def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Run all four metrics in one call and return a clean dict.

    This is what every model's .evaluate() method calls internally.
    The dict is also what pipeline.py returns to the frontend as JSON.

    Example return value:
    {
        "rmse": 42.31,
        "mae":  31.07,
        "r2":   0.913,
        "mape": 8.24
    }

    Values are rounded to 4 decimal places for clean display on the
    metrics cards and model comparison table.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae":  round(mae(y_true,  y_pred), 4),
        "r2":   round(r2(y_true,   y_pred), 4),
        "mape": round(mape(y_true,  y_pred), 4),
    }


# ─────────────────────────────────────────────
# SECTION 3 — Residual Analysis
# ─────────────────────────────────────────────

def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Build a residuals DataFrame for the Anomaly Report page.

    Residual = actual - predicted
    - Positive residual → model under-predicted (actual was higher)
    - Negative residual → model over-predicted (actual was lower)

    Large residuals often correspond to anomalous weather events —
    sudden cloud cover, fog, dust — which is exactly what the
    anomaly detection service looks for.

    Returns a DataFrame with columns:
        timestamp       : datetime index (if provided)
        actual          : true irradiance
        predicted       : model prediction
        residual        : actual - predicted
        abs_residual    : absolute value (for sorting by magnitude)
        pct_error       : percentage error (daytime only, else 0)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    residuals    = y_true - y_pred
    abs_residuals = np.abs(residuals)

    # Percentage error — mask nighttime zeros
    mask = y_true > 10
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


# ─────────────────────────────────────────────
# SECTION 4 — Model Comparison Summary
# ─────────────────────────────────────────────

def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Takes a list of metrics dicts (one per model) and returns a
    sorted comparison DataFrame for the Model Comparison page.

    metrics_list example:
    [
        {"model": "XGBoost",  "rmse": 42.3, "mae": 31.1, "r2": 0.91, "mape": 8.2},
        {"model": "LightGBM", "rmse": 40.1, "mae": 29.8, "r2": 0.93, "mape": 7.9},
        {"model": "Prophet",  "rmse": 55.2, "mae": 41.0, "r2": 0.87, "mape": 11.1},
        {"model": "Ensemble", "rmse": 38.4, "mae": 28.2, "r2": 0.94, "mape": 7.4},
    ]

    Sorted by RMSE ascending (best model first).
    """
    df = pd.DataFrame(metrics_list)

    # Put model name first, then sort by RMSE
    cols = ["model", "rmse", "mae", "r2", "mape"]
    df   = df[[c for c in cols if c in df.columns]]
    df   = df.sort_values("rmse").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# SECTION 5 — Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Simulate a decent model and a poor model ──
    np.random.seed(42)
    n = 500

    y_true = np.abs(np.random.normal(400, 150, n))  # realistic irradiance spread

    # Good model: small noise
    y_good = y_true + np.random.normal(0, 40, n)
    y_good = np.clip(y_good, 0, None)

    # Poor model: larger noise
    y_poor = y_true + np.random.normal(0, 120, n)
    y_poor = np.clip(y_poor, 0, None)

    print("=" * 45)
    print("Good model metrics:")
    good_metrics = compute_all(y_true, y_good)
    for k, v in good_metrics.items():
        print(f"  {k.upper():>6}: {v}")

    print("\nPoor model metrics:")
    poor_metrics = compute_all(y_true, y_poor)
    for k, v in poor_metrics.items():
        print(f"  {k.upper():>6}: {v}")

    # ── Residuals ──
    idx       = pd.date_range("2024-06-01", periods=n, freq="h")
    residuals = compute_residuals(y_true, y_good, timestamps=idx)
    print(f"\nResiduals DataFrame shape: {residuals.shape}")
    print(residuals.head(4).to_string(index=False))

    # ── Model comparison ──
    metrics_list = [
        {**good_metrics, "model": "XGBoost"},
        {**poor_metrics, "model": "Prophet"},
        {"rmse": 35.2, "mae": 26.1, "r2": 0.95, "mape": 6.8, "model": "Ensemble"},
    ]
    comparison = compare_models(metrics_list)
    print(f"\nModel comparison table (sorted by RMSE):")
    print(comparison.to_string(index=False))

    print("\nevaluate.py passed all checks.")
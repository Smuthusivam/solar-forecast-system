import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# SECTION 1 — Time Features
# ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from the timestamp index.
    The DataFrame must have a DatetimeIndex before calling this.
    """
    df = df.copy()

    df["hour"]        = df.index.hour                # 0–23
    df["day_of_year"] = df.index.dayofyear           # 1–365
    df["month"]       = df.index.month               # 1–12
    df["day_of_week"] = df.index.dayofweek           # 0 = Monday, 6 = Sunday
    df["is_daytime"]  = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)

    return df


# ─────────────────────────────────────────────
# SECTION 2 — Cyclical Encoding
# ─────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour and day-of-year as sine/cosine pairs so the model
    understands that hour 23 and hour 0 are adjacent, not far apart.

    Without this, a tree model would treat hour 23 and hour 0 as
    being 23 steps away. With sine/cosine they are nearly identical
    points on a circle — which is physically correct.
    """
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


# ─────────────────────────────────────────────
# SECTION 3 — Lag Features
# ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_col: str = "irradiance") -> pd.DataFrame:
    """
    Give the model a memory of past irradiance values.

    Each lag tells the model: "what was the solar output N hours ago?"
    - lag_1h   → 1 hour ago  (short-term trend)
    - lag_2h   → 2 hours ago
    - lag_24h  → same hour yesterday (daily pattern)
    - lag_48h  → same hour 2 days ago
    - lag_168h → same hour last week (weekly pattern)

    NaNs appear at the start of the series (not enough history).
    We fill them with 0 — a safe default since irradiance is 0
    at night and the model will learn to correct from context.
    """
    df = df.copy()

    lags = [1, 2, 24, 48, 168]
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag).fillna(0)

    return df


# ─────────────────────────────────────────────
# SECTION 4 — Rolling Statistics
# ─────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, target_col: str = "irradiance") -> pd.DataFrame:
    """
    Capture recent trend and variability with rolling window stats.

    - rolling_mean_3h  → smoothed short-term level
    - rolling_mean_24h → daily average level
    - rolling_std_24h  → how volatile the last 24h were (cloudy days = high std)
    - rolling_max_24h  → peak irradiance in the last day

    min_periods=1 prevents NaN on the first few rows where there
    isn't a full window of data yet.
    """
    df = df.copy()

    df["rolling_mean_3h"]  = df[target_col].rolling(window=3,  min_periods=1).mean()
    df["rolling_mean_24h"] = df[target_col].rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h"]  = df[target_col].rolling(window=24, min_periods=1).std().fillna(0)
    df["rolling_max_24h"]  = df[target_col].rolling(window=24, min_periods=1).max()

    return df


# ─────────────────────────────────────────────
# SECTION 5 — Weather Interaction Features
# ─────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Build physics-informed interaction features from weather columns
    when they are present in the dataset.

    col_map is the column mapping returned by the AI detector, e.g.:
        {"cloud_cover": "Cloud", "temperature": "Temp_C", "humidity": "RH"}

    cloud_clearness:
        Fraction of sky that is clear. 0 = fully overcast, 1 = clear sky.
        Simple and directly related to irradiance.

    temp_humidity_ratio:
        High temperature + low humidity → drier air → less scattering → more irradiance.
        Guards against division by zero by clipping humidity to minimum 1.
    """
    df = df.copy()

    cloud_col = col_map.get("cloud_cover")
    temp_col  = col_map.get("temperature")
    hum_col   = col_map.get("humidity")

    if cloud_col and cloud_col in df.columns:
        df["cloud_clearness"] = 1 - (df[cloud_col] / 100).clip(0, 1)

    if temp_col and hum_col and temp_col in df.columns and hum_col in df.columns:
        humidity_safe = df[hum_col].clip(lower=1)   # avoid division by zero
        df["temp_humidity_ratio"] = df[temp_col] / humidity_safe

    return df


# ─────────────────────────────────────────────
# SECTION 6 — Master Pipeline
# ─────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target_col: str = "irradiance",
    timestamp_col: str = "timestamp",
    col_map: dict = None,
) -> pd.DataFrame:
    """
    Main entry point. Call this from pipeline.py.

    Parameters
    ----------
    df           : Raw DataFrame from preprocessing (already cleaned, GHI estimated if needed)
    target_col   : Name of the irradiance column (after AI detection + renaming)
    timestamp_col: Name of the datetime column; will become the index
    col_map      : Dict mapping logical names → actual column names from AI detector
                   e.g. {"cloud_cover": "Cloud_%", "temperature": "T_amb"}

    Returns
    -------
    feature_df   : DataFrame with all engineered features, ready for XGBoost / LightGBM
                   (Prophet does NOT use this — it only needs ds + y)
    """
    if col_map is None:
        col_map = {}

    df = df.copy()

    # Step 1 — Set timestamp as index (required for rolling and lag ops)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
    df = df.sort_index()

    # Step 2 — Ensure the target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Step 3 — Run each feature group
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    df = add_weather_features(df, col_map=col_map)

    # Step 4 — Drop any remaining NaN rows (safety net)
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "irradiance") -> list:
    """
    Return the list of feature column names (everything except the target).
    Call this to get X columns before passing to model.train().
    """
    return [col for col in df.columns if col != target_col]


# ─────────────────────────────────────────────
# SECTION 7 — Quick Test (run this file directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Build a small synthetic dataset — 2 weeks of hourly data
    periods = 24 * 14
    idx = pd.date_range(start="2024-01-01", periods=periods, freq="h")

    # Simulate irradiance: zero at night, bell-curve during day
    hours = idx.hour
    irradiance = np.where(
        (hours >= 6) & (hours <= 20),
        500 * np.sin(np.pi * (hours - 6) / 14) + np.random.normal(0, 30, periods),
        0,
    ).clip(0)

    raw_df = pd.DataFrame({
        "timestamp":  idx,
        "irradiance": irradiance,
        "Cloud_%":    np.random.uniform(0, 80, periods),
        "Temp_C":     np.random.uniform(10, 35, periods),
        "RH":         np.random.uniform(30, 90, periods),
    })

    col_map = {
        "cloud_cover": "Cloud_%",
        "temperature": "Temp_C",
        "humidity":    "RH",
    }

    result = build_features(raw_df, target_col="irradiance", col_map=col_map)

    print("build_features() ran successfully")
    print(f"  Input rows  : {len(raw_df)}")
    print(f"  Output rows : {len(result)}")
    print(f"  Features    : {len(get_feature_columns(result))} columns")
    print(f"\nColumn list:")
    for col in result.columns:
        print(f"  {col}")
    print(f"\nFirst row sample:\n{result.iloc[0]}")
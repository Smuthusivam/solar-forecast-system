import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract hour, day of year, month, day of week, and a daytime flag from the index.
    df = df.copy()

    df["hour"]        = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["month"]       = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_daytime"]  = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encode hour and day-of-year as sine/cosine so the model knows 23:00 and 00:00 are adjacent.
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "irradiance") -> pd.DataFrame:
    # Add lagged irradiance values so the model can see what happened 1h, 24h, and 168h ago.
    df = df.copy()

    # Only use lags that can be safely computed without excessive NaN loss.
    # For datasets with < 400 rows, only use shorter lags to avoid losing data to dropna().
    lags = [1, 2, 24, 48, 168]
    if len(df) < 400:
        lags = [1, 2, 24]
        if len(df) < 200:
            lags = [1, 2, 24]  # Still try these; fillna(0) handles missing values
    
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag).fillna(0)

    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "irradiance") -> pd.DataFrame:
    # Add rolling mean, std, and max to capture recent trend and variability.
    df = df.copy()

    df["rolling_mean_3h"]  = df[target_col].rolling(window=3,  min_periods=1).mean()
    df["rolling_mean_24h"] = df[target_col].rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h"]  = df[target_col].rolling(window=24, min_periods=1).std().fillna(0)
    df["rolling_max_24h"]  = df[target_col].rolling(window=24, min_periods=1).max().fillna(0)

    return df


def add_weather_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    # Build physics-informed interaction features when weather columns are present.
    # After preprocessing, columns are already renamed to standard names, so we
    # check both the standard name directly AND via col_map for flexibility.
    df = df.copy()

    def _find_col(standard_name: str) -> str | None:
        if standard_name in df.columns:
            return standard_name
        mapped = col_map.get(standard_name)
        if mapped and mapped in df.columns:
            return mapped
        return None

    cloud_col = _find_col("cloud_cover")
    temp_col  = _find_col("temperature")
    hum_col   = _find_col("humidity")

    if cloud_col:
        df["cloud_clearness"] = 1 - (df[cloud_col] / 100).clip(0, 1)

    if temp_col and hum_col:
        humidity_safe = df[hum_col].clip(lower=1)
        df["temp_humidity_ratio"] = df[temp_col] / humidity_safe

    return df


def build_features(
    df:            pd.DataFrame,
    target_col:    str  = "irradiance",
    timestamp_col: str  = "timestamp",
    col_map:       dict = None,
) -> pd.DataFrame:
    """
    Main entry point — call this from pipeline.py to get a model-ready feature DataFrame.

    Args:
        df:            Clean DataFrame from preprocessing with 'irradiance' column.
        target_col:    Name of the irradiance column after AI detection and renaming.
        timestamp_col: Datetime column name; becomes the index if not already set.
        col_map:       Dict mapping logical names to actual column names from the AI detector.

    Returns a DataFrame with all engineered features ready for XGBoost and LightGBM.
    Prophet does NOT use this — it only needs ds + y.
    """
    if col_map is None:
        col_map = {}

    df = df.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
    df = df.sort_index()

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    df = add_weather_features(df, col_map=col_map)

    # Only drop NA rows if we still have sufficient data; this prevents losing data
    # when predicting on shorter sequences or corrected datasets.
    rows_before = len(df)
    na_count = df.isna().sum().sum()
    
    if na_count > 0:
        rows_after_dropna = len(df.dropna())
        # If dropna() removes >50% of rows, fillna instead (lag-induced NaNs)
        if rows_after_dropna < max(1, rows_before * 0.5):
            df = df.fillna(0)
        else:
            df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "irradiance") -> list:
    """Return all column names except the target — these are the model inputs."""
    return [col for col in df.columns if col != target_col]


if __name__ == "__main__":
    periods = 24 * 14
    idx     = pd.date_range(start="2024-01-01", periods=periods, freq="h")
    hours   = idx.hour

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

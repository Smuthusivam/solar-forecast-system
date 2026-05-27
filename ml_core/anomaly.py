# Anomaly detection for solar irradiance — three statistical methods + physical validation filters.

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYTIME_START = 6
DAYTIME_END   = 20

# Minimum irradiance to consider for anomaly detection (ignore near-zero dawn/dusk readings)
MIN_IRRADIANCE = 50.0

# Statistical thresholds
ZSCORE_LOW    = 2.5
ZSCORE_MEDIUM = 3.5
ZSCORE_HIGH   = 5.0

IQR_LOW    = 2.0
IQR_MEDIUM = 3.0
IQR_HIGH   = 4.0

ROLLING_LOW    = 0.85
ROLLING_MEDIUM = 0.92
ROLLING_HIGH   = 0.97

# Skip sunrise/sunset — natural large gradients, not anomalies
TRANSITION_HOURS = {6, 7, 8, 18, 19, 20}

ZSCORE_WINDOW = 24
IQR_WINDOW    = 24

# Maximum physically possible GHI by hour-of-day (W/m²)
# Based on extraterrestrial radiation × typical clear-sky transmittance (~0.75)
_HOURLY_MAX_GHI = {
    0: 0,   1: 0,   2: 0,   3: 0,   4: 0,
    5: 20,  6: 100, 7: 280, 8: 480, 9: 650,
    10: 780, 11: 870, 12: 900, 13: 870, 14: 800,
    15: 680, 16: 520, 17: 340, 18: 160, 19: 50,
    20: 10, 21: 0,  22: 0,  23: 0,
}


# Physical validation filters — applied AFTER statistical detection


def _hourly_max(hour: int) -> float:
    return float(_HOURLY_MAX_GHI.get(hour, 900))


def _is_physically_impossible(value: float, hour: int) -> bool:
    """True if the value exceeds what is physically possible at this hour."""
    return value > _hourly_max(hour) * 1.10  # 10% tolerance for measurement uncertainty


def _is_weather_coherent(value: float, ts: pd.Timestamp, df: pd.DataFrame) -> bool:
    """
    Return True (coherent = NOT an anomaly) when available weather columns
    support the irradiance reading.
    - High cloud cover + high irradiance → incoherent → anomaly
    - Low cloud cover + low irradiance at midday → incoherent → anomaly
    - No weather columns → always coherent (can't judge)
    """
    if "cloud_cover" in df.columns:
        try:
            cloud = float(df.at[ts, "cloud_cover"])
            hour  = ts.hour

            # Very high cloud cover (>80%) but bright sunshine → suspicious
            if cloud > 80 and value > 600:
                return False

            # Clear sky (cloud <10%) but near-zero at solar peak → suspicious
            if cloud < 10 and 10 <= hour <= 14 and value < 100:
                return False
        except Exception:
            pass

    return True  # no weather data or no conflict




# Severity helpers

def _severity_from_zscore(z: float) -> str:
    if abs(z) >= ZSCORE_HIGH:   return "high"
    if abs(z) >= ZSCORE_MEDIUM: return "medium"
    return "low"


def _severity_from_iqr(distance: float, iqr: float) -> str:
    if iqr == 0: return "low"
    ratio = distance / iqr
    if ratio >= IQR_HIGH:   return "high"
    if ratio >= IQR_MEDIUM: return "medium"
    return "low"


def _severity_from_change(pct: float) -> str:
    if pct >= ROLLING_HIGH:   return "high"
    if pct >= ROLLING_MEDIUM: return "medium"
    return "low"


def _is_daytime(hour: int) -> bool:
    return DAYTIME_START <= hour <= DAYTIME_END


# Statistical detectors

def _detect_zscore(series: pd.Series) -> list[dict]:
    rolling_mean = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).mean()
    rolling_std  = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).std()
    anomalies    = []

    for ts, value in series.items():
        if not _is_daytime(ts.hour): continue
        if value < MIN_IRRADIANCE:   continue

        mean = rolling_mean[ts]
        std  = rolling_std[ts]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue

        z = (value - mean) / std
        if abs(z) >= ZSCORE_LOW:
            anomalies.append({
                "timestamp": ts,
                "value":     round(float(value), 2),
                "expected":  round(float(mean),  2),
                "deviation": round(float(abs(value - mean)), 2),
                "severity":  _severity_from_zscore(z),
                "method":    "zscore",
                "score":     round(float(abs(z)), 3),
            })

    logger.info("Z-score: %d raw anomalies", len(anomalies))
    return anomalies


def _detect_iqr(series: pd.Series) -> list[dict]:
    anomalies = []

    for i, (ts, value) in enumerate(series.items()):
        if not _is_daytime(ts.hour): continue
        if value < MIN_IRRADIANCE:   continue

        start       = max(0, i - IQR_WINDOW)
        window_vals = series.iloc[start:i].values
        if len(window_vals) < 6:
            continue

        q1  = np.percentile(window_vals, 25)
        q3  = np.percentile(window_vals, 75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - IQR_LOW * iqr
        upper_bound = q3 + IQR_LOW * iqr

        if value < lower_bound or value > upper_bound:
            expected  = (q1 + q3) / 2.0
            deviation = abs(value - expected)
            anomalies.append({
                "timestamp": ts,
                "value":     round(float(value),    2),
                "expected":  round(float(expected), 2),
                "deviation": round(float(deviation),2),
                "severity":  _severity_from_iqr(deviation, iqr),
                "method":    "iqr",
                "score":     round(float(deviation / iqr), 3),
            })

    logger.info("IQR: %d raw anomalies", len(anomalies))
    return anomalies


def _detect_rolling(series: pd.Series) -> list[dict]:
    pct_change = series.pct_change().abs()
    anomalies  = []

    for ts, value in series.items():
        if not _is_daytime(ts.hour): continue
        if value < MIN_IRRADIANCE:   continue
        if ts.hour in TRANSITION_HOURS: continue

        change = pct_change[ts]
        if pd.isna(change):
            continue

        prev_value = series.shift(1)[ts]
        if pd.isna(prev_value) or prev_value < MIN_IRRADIANCE:
            continue

        if change >= ROLLING_LOW:
            anomalies.append({
                "timestamp": ts,
                "value":     round(float(value),      2),
                "expected":  round(float(prev_value), 2),
                "deviation": round(abs(float(value) - float(prev_value)), 2),
                "severity":  _severity_from_change(change),
                "method":    "rolling",
                "score":     round(float(change), 3),
            })

    logger.info("Rolling: %d raw anomalies", len(anomalies))
    return anomalies


# Merge + physical filter pass


def _merge_anomalies(all_anomalies: list[dict]) -> list[dict]:
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    by_timestamp: dict[datetime, dict] = {}

    for anomaly in all_anomalies:
        ts = anomaly["timestamp"]
        if ts not in by_timestamp:
            by_timestamp[ts] = anomaly
        else:
            existing = by_timestamp[ts]
            if severity_rank[anomaly["severity"]] > severity_rank[existing["severity"]]:
                by_timestamp[ts] = anomaly
            elif (
                severity_rank[anomaly["severity"]] == severity_rank[existing["severity"]]
                and anomaly["score"] > existing.get("score", 0)
            ):
                by_timestamp[ts] = anomaly

    merged = sorted(by_timestamp.values(), key=lambda x: x["timestamp"])
    for a in merged:
        a.pop("score", None)
    return merged


def _apply_physical_filters(anomalies: list[dict], df: pd.DataFrame) -> list[dict]:
    """
    Remove anomalies that are genuinely explainable by physics or weather:
    1. Only dismiss if value is physically impossible AND weather-incoherent together
       (i.e. dismiss only clear false positives, not borderline outliers)
    2. Only dismiss consecutive runs of 6+ points (very extended weather events);
       shorter runs of 3-5 may still be sensor faults
    Returns only genuinely suspicious readings.
    """
    timestamps = [a["timestamp"] for a in anomalies]
    # Raise the consecutive run threshold so shorter spikes are still flagged
    weather_event_ts = set()
    if len(timestamps) >= 6:
        sorted_ts = sorted(timestamps)
        run: list = [sorted_ts[0]]
        for i in range(1, len(sorted_ts)):
            gap = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 3600
            if gap <= 1.5:
                run.append(sorted_ts[i])
            else:
                if len(run) >= 6:
                    weather_event_ts.update(run)
                run = [sorted_ts[i]]
        if len(run) >= 6:
            weather_event_ts.update(run)

    filtered  = []
    dismissed = 0

    for a in anomalies:
        ts    = a["timestamp"]
        value = a["value"]
        hour  = ts.hour

        # Dismiss only if physically impossible (exceeds hourly ceiling by >10%)
        if _is_physically_impossible(value, hour):
            # Keep — physically impossible readings are genuine anomalies
            filtered.append(a)
            continue

        # Dismiss only if weather explicitly contradicts the reading AND
        # the anomaly severity is low (high/medium get through regardless)
        if not _is_weather_coherent(value, ts, df) and a["severity"] == "low":
            dismissed += 1
            continue

        # Dismiss only very long consecutive runs (6+) as weather events
        if ts in weather_event_ts:
            dismissed += 1
            continue

        filtered.append(a)

    logger.info(
        "Physical filter: %d dismissed (%d long weather runs, %d weather-incoherent low), %d remain",
        dismissed,
        len(weather_event_ts & set(timestamps)),
        dismissed - len(weather_event_ts & set(timestamps)),
        len(filtered),
    )
    return filtered


# Public entry point

def detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    try:
        series = df["irradiance"].copy()

        logger.info(
            "Anomaly detection started | %d total points | %s → %s",
            len(series), series.index[0], series.index[-1],
        )

        all_anomalies = (
            _detect_zscore(series)
            + _detect_iqr(series)
            + _detect_rolling(series)
        )
        merged   = _merge_anomalies(all_anomalies)
        filtered = _apply_physical_filters(merged, df)

        total_points  = len(series)
        anomaly_count = len(filtered)
        anomaly_rate  = round(anomaly_count / total_points, 4) if total_points > 0 else 0.0

        logger.info(
            "Anomaly detection complete | %d genuine anomalies / %d points (%.1f%%)",
            anomaly_count, total_points, anomaly_rate * 100,
        )

        for a in filtered:
            a["timestamp"] = a["timestamp"].isoformat()

        return {
            "total_points":  total_points,
            "anomaly_count": anomaly_count,
            "anomaly_rate":  anomaly_rate,
            "anomalies":     filtered,
        }

    except Exception as exc:
        logger.error("Anomaly detection failed: %s", exc, exc_info=True)
        return {
            "total_points":  len(df) if df is not None else 0,
            "anomaly_count": 0,
            "anomaly_rate":  0.0,
            "anomalies":     [],
        }

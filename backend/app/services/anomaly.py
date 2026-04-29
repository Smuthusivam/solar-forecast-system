# Anomaly detection for solar irradiance — three methods run independently and are merged.

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYTIME_START = 6
DAYTIME_END   = 20

# Only flag daytime points above this threshold — ignores nighttime noise and dawn/dusk
MIN_IRRADIANCE = 100.0

ZSCORE_LOW    = 2.5
ZSCORE_MEDIUM = 3.5
ZSCORE_HIGH   = 5.0

IQR_LOW    = 2.0
IQR_MEDIUM = 3.0
IQR_HIGH   = 4.0

ROLLING_LOW    = 0.80
ROLLING_MEDIUM = 0.90
ROLLING_HIGH   = 0.97

# Skip sunrise/sunset hours in rolling detection — natural large changes, not anomalies
TRANSITION_HOURS = {6, 7, 8, 18, 19, 20}

ZSCORE_WINDOW = 24
IQR_WINDOW    = 24


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


def _detect_zscore(series: pd.Series) -> list[dict]:
    # Flag daytime points that deviate more than ZSCORE_LOW standard deviations from the rolling mean.
    rolling_mean = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).mean()
    rolling_std  = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).std()

    anomalies = []

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
                "value":     round(float(value),        2),
                "expected":  round(float(mean),         2),
                "deviation": round(float(abs(value - mean)), 2),
                "severity":  _severity_from_zscore(z),
                "method":    "zscore",
                "score":     round(float(abs(z)),       3),
            })

    logger.info("Z-score: %d anomalies detected", len(anomalies))
    return anomalies


def _detect_iqr(series: pd.Series) -> list[dict]:
    # Flag points outside IQR_LOW × interquartile range — more robust than Z-score against outliers.
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
                "value":     round(float(value),     2),
                "expected":  round(float(expected),  2),
                "deviation": round(float(deviation), 2),
                "severity":  _severity_from_iqr(deviation, iqr),
                "method":    "iqr",
                "score":     round(float(deviation / iqr), 3),
            })

    logger.info("IQR: %d anomalies detected", len(anomalies))
    return anomalies


def _detect_rolling(series: pd.Series) -> list[dict]:
    # Flag mid-day points where irradiance changes by more than ROLLING_LOW in one hour.
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
            expected  = float(prev_value)
            deviation = abs(float(value) - expected)

            anomalies.append({
                "timestamp": ts,
                "value":     round(float(value), 2),
                "expected":  round(expected,      2),
                "deviation": round(deviation,     2),
                "severity":  _severity_from_change(change),
                "method":    "rolling",
                "score":     round(float(change), 3),
            })

    logger.info("Rolling window: %d anomalies detected", len(anomalies))
    return anomalies


def _merge_anomalies(all_anomalies: list[dict]) -> list[dict]:
    # Combine results from all three methods; keep the highest severity per timestamp.
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


def detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    # Run all three detection methods, merge results, and return a summary dict.
    try:
        series = df["irradiance"].copy()

        logger.info(
            "Anomaly detection started | %d total points | %s → %s",
            len(series), series.index[0], series.index[-1],
        )

        zscore_anomalies  = _detect_zscore(series)
        iqr_anomalies     = _detect_iqr(series)
        rolling_anomalies = _detect_rolling(series)

        all_anomalies = zscore_anomalies + iqr_anomalies + rolling_anomalies
        merged        = _merge_anomalies(all_anomalies)

        total_points  = len(series)
        anomaly_count = len(merged)
        anomaly_rate  = round(anomaly_count / total_points, 4) if total_points > 0 else 0.0

        logger.info(
            "Anomaly detection complete | %d anomalies / %d points (%.1f%%)",
            anomaly_count, total_points, anomaly_rate * 100,
        )

        # Serialise timestamps to ISO strings for the JSON response
        for a in merged:
            a["timestamp"] = a["timestamp"].isoformat()

        return {
            "total_points":  total_points,
            "anomaly_count": anomaly_count,
            "anomaly_rate":  anomaly_rate,
            "anomalies":     merged,
        }

    except Exception as exc:
        logger.error("Anomaly detection failed: %s", exc)
        return {
            "total_points":  len(df) if df is not None else 0,
            "anomaly_count": 0,
            "anomaly_rate":  0.0,
            "anomalies":     [],
        }

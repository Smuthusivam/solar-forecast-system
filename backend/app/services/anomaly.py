"""
anomaly.py — Anomaly detection for solar irradiance time series.

Three detection methods run independently and results are merged:

  1. Z-score        — flags values > 2.5 standard deviations from rolling mean
  2. IQR            — flags values outside 2.0× interquartile range
  3. Rolling window — flags sudden changes > 70% in 1 hour (mid-day only)

Tuning decisions:
  - MIN_IRRADIANCE = 50 W/m²  → ignores nighttime noise and dawn/dusk transitions
  - ROLLING skips hours 6,7,19,20 → ignores natural sunrise/sunset ramps
  - Higher thresholds throughout → targets real anomalies, not sensor noise
  - Expected anomaly rate: 1–5% of daytime points
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DAYTIME_START   = 6
DAYTIME_END     = 20

# Raised from 10 → 50 W/m²
# Ignores dawn/dusk transitions and nighttime noise completely
MIN_IRRADIANCE  = 100.0

# Z-score thresholds — raised to reduce false positives
ZSCORE_LOW      = 2.5    # was 2.0
ZSCORE_MEDIUM   = 3.5    # was 3.0
ZSCORE_HIGH     = 5.0    # was 4.0

# IQR multipliers — more conservative
IQR_LOW         = 2.0    # was 1.5
IQR_MEDIUM      = 3.0    # was 2.5
IQR_HIGH        = 4.0    # was 3.5

# Rolling window — raised from 40% to 70%
# Natural sunrise goes 0 → 500 W/m² — that's NOT anomalous
# Only flag truly sudden mid-day changes (cloud events, sensor faults)
ROLLING_LOW     = 0.80   # was 0.40
ROLLING_MEDIUM  = 0.90   # was 0.60
ROLLING_HIGH    = 0.97   # was 0.80

# Transition hours to skip in rolling detection
# Hours 6,7 = sunrise ramp up | Hours 19,20 = sunset ramp down
TRANSITION_HOURS = {6, 7,8 , 18, 19, 20}

# Rolling window sizes
ZSCORE_WINDOW   = 24
IQR_WINDOW      = 24


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_from_zscore(z: float) -> str:
    if abs(z) >= ZSCORE_HIGH:
        return "high"
    if abs(z) >= ZSCORE_MEDIUM:
        return "medium"
    return "low"


def _severity_from_iqr(distance: float, iqr: float) -> str:
    if iqr == 0:
        return "low"
    ratio = distance / iqr
    if ratio >= IQR_HIGH:
        return "high"
    if ratio >= IQR_MEDIUM:
        return "medium"
    return "low"


def _severity_from_change(pct: float) -> str:
    if pct >= ROLLING_HIGH:
        return "high"
    if pct >= ROLLING_MEDIUM:
        return "medium"
    return "low"


def _is_daytime(hour: int) -> bool:
    return DAYTIME_START <= hour <= DAYTIME_END


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — Z-score
# ─────────────────────────────────────────────────────────────────────────────

def _detect_zscore(series: pd.Series) -> list[dict]:
    """
    Flag points where irradiance deviates more than ZSCORE_LOW
    standard deviations from the rolling mean.

    Only evaluates daytime points with irradiance > MIN_IRRADIANCE.
    """
    rolling_mean = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).mean()
    rolling_std  = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).std()

    anomalies = []

    for ts, value in series.items():
        if not _is_daytime(ts.hour):
            continue
        if value < MIN_IRRADIANCE:
            continue

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


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — IQR
# ─────────────────────────────────────────────────────────────────────────────

def _detect_iqr(series: pd.Series) -> list[dict]:
    """
    Flag points outside IQR_LOW × interquartile range.

    More robust than Z-score — not affected by extreme outliers
    in the reference window.
    """
    anomalies = []

    for i, (ts, value) in enumerate(series.items()):
        if not _is_daytime(ts.hour):
            continue
        if value < MIN_IRRADIANCE:
            continue

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


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — Rolling window (sudden change detection)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_rolling(series: pd.Series) -> list[dict]:
    """
    Flag points where irradiance changes by more than ROLLING_LOW %
    compared to the previous hour.

    Skips:
      - Nighttime hours
      - Points below MIN_IRRADIANCE
      - Transition hours (6,7 = sunrise | 19,20 = sunset)
        These are natural large changes, not anomalies.

    Catches: cloud events, sudden shading, sensor faults mid-day.
    """
    pct_change = series.pct_change().abs()
    anomalies  = []

    for ts, value in series.items():
        if not _is_daytime(ts.hour):
            continue
        if value < MIN_IRRADIANCE:
            continue

        # Skip sunrise/sunset transition hours — natural large changes
        if ts.hour in TRANSITION_HOURS:
            continue

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
                "value":     round(float(value),  2),
                "expected":  round(expected,       2),
                "deviation": round(deviation,      2),
                "severity":  _severity_from_change(change),
                "method":    "rolling",
                "score":     round(float(change),  3),
            })

    logger.info("Rolling window: %d anomalies detected", len(anomalies))
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Merge + deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def _merge_anomalies(all_anomalies: list[dict]) -> list[dict]:
    """
    Merge results from all three methods.
    When multiple methods flag the same timestamp, keep the highest severity.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all three anomaly detection methods and return merged results.

    Args:
        df: Clean DataFrame with DatetimeIndex and 'irradiance' column.

    Returns:
        {
            "total_points":  int,
            "anomaly_count": int,
            "anomaly_rate":  float,
            "anomalies":     list of anomaly dicts
        }

    Never raises — returns empty list on any error.
    """
    try:
        series = df["irradiance"].copy()

        logger.info(
            "Anomaly detection started | %d total points | %s → %s",
            len(series), series.index[0], series.index[-1],
        )

        # Run all three methods
        zscore_anomalies  = _detect_zscore(series)
        iqr_anomalies     = _detect_iqr(series)
        rolling_anomalies = _detect_rolling(series)

        # Merge and deduplicate
        all_anomalies = zscore_anomalies + iqr_anomalies + rolling_anomalies
        merged        = _merge_anomalies(all_anomalies)

        total_points  = len(series)
        anomaly_count = len(merged)
        anomaly_rate  = round(anomaly_count / total_points, 4) if total_points > 0 else 0.0

        logger.info(
            "Anomaly detection complete | %d anomalies / %d points (%.1f%%)",
            anomaly_count, total_points, anomaly_rate * 100,
        )

        # Serialise timestamps to ISO strings for JSON response
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
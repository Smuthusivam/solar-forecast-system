"""
anomaly.py — Anomaly detection for solar irradiance time series.

Three detection methods run independently and results are merged:

  1. Z-score        — flags values > 3 standard deviations from rolling mean
                      Best for: gradual sensor drift, systematic offsets
  
  2. IQR            — flags values outside 1.5× interquartile range
                      Best for: robust detection that ignores extreme outliers
                      in the reference window (more stable than Z-score)

  3. Rolling window — flags sudden spikes/drops > 50% change in 1 hour
                      Best for: cloud events, sudden shading, sensor faults

Each method assigns severity (low / medium / high) based on how far
the value deviates from expected. The final output merges all three,
deduplicates overlapping detections, and returns a clean list sorted
by timestamp.

Only daytime points are evaluated (hour 6–20) — nighttime irradiance
is always ~0 so any detection there is meaningless noise.
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

DAYTIME_START   = 6     # hour (inclusive)
DAYTIME_END     = 20    # hour (inclusive)

# Z-score thresholds
ZSCORE_LOW      = 2.0
ZSCORE_MEDIUM   = 3.0
ZSCORE_HIGH     = 4.0

# IQR multiplier thresholds
IQR_LOW         = 1.5
IQR_MEDIUM      = 2.5
IQR_HIGH        = 3.5

# Rolling window — minimum % change to flag
ROLLING_LOW     = 0.40   # 40% change
ROLLING_MEDIUM  = 0.60   # 60% change
ROLLING_HIGH    = 0.80   # 80% change

# Minimum irradiance to consider a point for anomaly detection
MIN_IRRADIANCE  = 10.0   # W/m² — ignore near-zero readings

# Rolling window sizes
ZSCORE_WINDOW   = 24     # hours — rolling mean/std reference window
IQR_WINDOW      = 24     # hours


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
    Flag points where the irradiance value deviates more than ZSCORE_LOW
    standard deviations from the rolling mean.

    Rolling mean/std computed over ZSCORE_WINDOW hours so the reference
    adapts to local conditions (cloudy day vs sunny day).
    """
    rolling_mean = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).mean()
    rolling_std  = series.rolling(window=ZSCORE_WINDOW, min_periods=3, center=True).std()

    anomalies = []

    for ts, value in series.items():
        # Skip nighttime and near-zero readings
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
                "value":     round(float(value),    2),
                "expected":  round(float(mean),     2),
                "deviation": round(float(abs(value - mean)), 2),
                "severity":  _severity_from_zscore(z),
                "method":    "zscore",
                "score":     round(float(abs(z)),   3),
            })

    logger.info("Z-score: %d anomalies detected", len(anomalies))
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — IQR
# ─────────────────────────────────────────────────────────────────────────────

def _detect_iqr(series: pd.Series) -> list[dict]:
    """
    Flag points outside 1.5× IQR from the rolling Q1/Q3 bounds.

    IQR is more robust than Z-score because it's not affected by
    extreme outliers in the reference window — Q1 and Q3 are resistant
    to a few very large or very small values.
    """
    anomalies = []

    for i, (ts, value) in enumerate(series.items()):
        if not _is_daytime(ts.hour):
            continue
        if value < MIN_IRRADIANCE:
            continue

        # Reference window: past IQR_WINDOW hours
        start = max(0, i - IQR_WINDOW)
        window_vals = series.iloc[start:i].values

        if len(window_vals) < 6:   # not enough history yet
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

    This catches cloud events and sensor faults that Z-score and IQR
    miss because the absolute value may still be within normal range —
    it's the *rate of change* that's anomalous.

    Example: 600 W/m² → 50 W/m² in one hour is a 92% drop — clearly
    a cloud event or equipment fault, even though both values are
    individually plausible.
    """
    pct_change = series.pct_change().abs()
    anomalies  = []

    for ts, value in series.items():
        if not _is_daytime(ts.hour):
            continue
        if value < MIN_IRRADIANCE:
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
                "value":     round(float(value),    2),
                "expected":  round(expected,         2),
                "deviation": round(deviation,        2),
                "severity":  _severity_from_change(change),
                "method":    "rolling",
                "score":     round(float(change),   3),
            })

    logger.info("Rolling window: %d anomalies detected", len(anomalies))
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Merge + deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def _merge_anomalies(all_anomalies: list[dict]) -> list[dict]:
    """
    Merge results from all three methods.

    When multiple methods flag the same timestamp, keep the one with
    the highest severity. If severity is equal, prefer the one with
    the highest score (most extreme deviation).

    Severity ranking: high > medium > low
    """
    severity_rank = {"high": 3, "medium": 2, "low": 1}

    # Group by timestamp
    by_timestamp: dict[datetime, dict] = {}

    for anomaly in all_anomalies:
        ts = anomaly["timestamp"]

        if ts not in by_timestamp:
            by_timestamp[ts] = anomaly
        else:
            existing = by_timestamp[ts]
            # Keep the higher severity
            if severity_rank[anomaly["severity"]] > severity_rank[existing["severity"]]:
                by_timestamp[ts] = anomaly
            elif (
                severity_rank[anomaly["severity"]] == severity_rank[existing["severity"]]
                and anomaly["score"] > existing.get("score", 0)
            ):
                by_timestamp[ts] = anomaly

    # Sort by timestamp and clean up internal score field
    merged = sorted(by_timestamp.values(), key=lambda x: x["timestamp"])

    for a in merged:
        a.pop("score", None)   # internal field — not in API response

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all three anomaly detection methods and return merged results.

    Args:
        df: Clean DataFrame with DatetimeIndex and 'irradiance' column.
            Produced by preprocessing.preprocess().

    Returns:
        {
            "total_points":  int,
            "anomaly_count": int,
            "anomaly_rate":  float,
            "anomalies": [
                {
                    "timestamp": datetime,
                    "value":     float,   # observed W/m²
                    "expected":  float,   # rolling reference W/m²
                    "deviation": float,   # abs difference
                    "severity":  str,     # "low" | "medium" | "high"
                    "method":    str,     # "zscore" | "iqr" | "rolling"
                },
                ...
            ]
        }

    Never raises — returns empty anomaly list on any error.
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
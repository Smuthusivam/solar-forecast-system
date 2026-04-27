"""
preprocessing.py — Data cleaning, validation, and GHI estimation.

Responsibilities:
  1. Parse and validate the uploaded CSV
  2. Standardise column names using the AI detector mapping
  3. Clean the data (missing values, outliers, duplicates)
  4. Parse timestamps into a proper DatetimeIndex
  5. Estimate GHI from weather variables if no irradiance column exists
     (Angstrom-Prescott physics-informed formula)
  6. Return a clean DataFrame ready for feature_engineering.py

Two execution modes:
  direct    → irradiance column found, clean and use it directly
  estimated → no irradiance column, compute GHI from weather variables

The output DataFrame always has these standardised columns:
  - timestamp     (DatetimeIndex)
  - irradiance    (W/m², float)
  - temperature   (°C, float)      if available
  - humidity      (%, float)       if available
  - wind_speed    (m/s, float)     if available
  - cloud_cover   (%, float)       if available
  - sunshine_hours (hours, float)  if available
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Physical bounds for irradiance — values outside are physically impossible
IRRADIANCE_MIN =    0.0   # W/m²  (can't be negative)
IRRADIANCE_MAX = 1400.0   # W/m²  (extraterrestrial solar constant ~1361 W/m²)

# Bounds for other weather variables
TEMP_MIN,  TEMP_MAX  = -50.0,   60.0   # °C
HUM_MIN,   HUM_MAX   =   0.0,  100.0   # %
WIND_MIN,  WIND_MAX  =   0.0,   75.0   # m/s
CLOUD_MIN, CLOUD_MAX =   0.0,  100.0   # %
SUN_MIN,   SUN_MAX   =   0.0,   24.0   # hours

# Minimum rows needed to train ML models reliably
MIN_ROWS = 48   # 2 days of hourly data

# Angstrom-Prescott solar constant at Earth's surface
SOLAR_CONSTANT = 1000.0   # W/m²


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse raw CSV bytes into a DataFrame.

    Tries common encodings and separators automatically so the user
    doesn't have to worry about file format details.

    Raises:
        ValueError if the file cannot be parsed or is empty.
    """
    encodings  = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1"]
    separators = [",", ";", "\t", "|"]

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep       = sep,
                    encoding  = encoding,
                    low_memory= False,
                )
                # Must have at least 2 columns and MIN_ROWS rows
                if df.shape[1] >= 2 and len(df) >= MIN_ROWS:
                    logger.info(
                        "CSV parsed: %d rows × %d cols (encoding=%s sep=%r)",
                        len(df), df.shape[1], encoding, sep,
                    )
                    return df
            except Exception:
                continue

    raise ValueError(
        f"Could not parse the CSV file. "
        f"Ensure it has at least {MIN_ROWS} rows and uses a standard format."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Column standardisation
# ─────────────────────────────────────────────────────────────────────────────

def standardise_columns(
    df:      pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    """
    Rename detected columns to standardised internal names.

    Example:
        mapping = {"irradiance": "GHI", "temperature": "Temp_C", ...}
        → renames "GHI" → "irradiance", "Temp_C" → "temperature"

    Columns not in the mapping are dropped — we only keep what we need.
    """
    # Reverse the mapping: original_name → standard_name
    rename_map = {
        original: standard
        for standard, original in mapping.items()
        if original is not None
    }

    # Keep only columns that appear in the mapping
    cols_to_keep = [c for c in df.columns if c in rename_map]
    df = df[cols_to_keep].rename(columns=rename_map)

    logger.info("Standardised columns: %s", list(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Timestamp parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'timestamp' column to a proper DatetimeIndex.
    Handles ANY format through 4 progressive layers — never crashes.
    """
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found — generating synthetic hourly index")
        df["timestamp"] = pd.date_range(
            start="2020-01-01",
            periods=len(df),
            freq="h",
        )
        df = df.set_index("timestamp").sort_index()
        return df

    parsed = None

    # ── Layer 1: pandas auto-detect ───────────────────────────────────────────
    try:
        parsed = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
        if parsed.isna().sum() / len(parsed) < 0.5:
            logger.info("Timestamps parsed via pandas auto-detect")
    except Exception:
        parsed = None

    # ── Layer 2: explicit common formats ──────────────────────────────────────
    if parsed is None or parsed.isna().all():
        common_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%Y%m%d%H%M%S",
            "%Y%m%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
        ]
        for fmt in common_formats:
            try:
                candidate = pd.to_datetime(df["timestamp"], format=fmt)
                if not candidate.isna().all():
                    parsed = candidate
                    logger.info("Timestamps parsed with explicit format: %s", fmt)
                    break
            except Exception:
                continue

    # ── Layer 3: dateutil parser (handles almost any human format) ────────────
    if parsed is None or parsed.isna().all():
        try:
            from dateutil import parser as dateutil_parser
            parsed = df["timestamp"].apply(
                lambda x: dateutil_parser.parse(str(x), fuzzy=True)
                if pd.notna(x) else pd.NaT
            )
            parsed = pd.to_datetime(parsed)
            logger.info("Timestamps parsed via dateutil fuzzy parser")
        except Exception as exc:
            logger.warning("dateutil parser failed: %s", exc)
            parsed = None

    # ── Layer 4: Unix timestamp (seconds or milliseconds) ─────────────────────
    if parsed is None or parsed.isna().all():
        try:
            numeric = pd.to_numeric(df["timestamp"], errors="coerce")
            if numeric.notna().sum() > len(df) * 0.5:
                # Detect milliseconds vs seconds
                if numeric.median() > 1e10:
                    parsed = pd.to_datetime(numeric, unit="ms")
                else:
                    parsed = pd.to_datetime(numeric, unit="s")
                logger.info("Timestamps parsed as Unix timestamp")
        except Exception:
            parsed = None

    # ── Layer 5: synthetic fallback — never crash ─────────────────────────────
    if parsed is None or parsed.isna().all():
        logger.warning(
            "Could not parse timestamps — generating synthetic hourly index"
        )
        parsed = pd.date_range(start="2020-01-01", periods=len(df), freq="h")

    df["timestamp"] = parsed

    # ── Set as index, sort, remove duplicates ─────────────────────────────────
    df = df.set_index("timestamp").sort_index()

    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Removed %d duplicate timestamps", n_dupes)
        df = df[~df.index.duplicated(keep="last")]

    logger.info(
        "Timestamp index: %s → %s (%d rows)",
        df.index[0], df.index[-1], len(df),
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. Type coercion + bounds cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-index columns to float, coercing errors to NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clip_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip physical variables to their valid ranges.
    Values outside the range are set to NaN (not clipped to boundary)
    because a value of -50 W/m² is not "close to 0" — it's a sensor error.
    """
    bounds = {
        "irradiance":     (IRRADIANCE_MIN, IRRADIANCE_MAX),
        "temperature":    (TEMP_MIN,  TEMP_MAX),
        "humidity":       (HUM_MIN,   HUM_MAX),
        "wind_speed":     (WIND_MIN,  WIND_MAX),
        "cloud_cover":    (CLOUD_MIN, CLOUD_MAX),
        "sunshine_hours": (SUN_MIN,   SUN_MAX),
    }

    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                logger.warning(
                    "Column '%s': %d values outside [%.1f, %.1f] → set to NaN",
                    col, n_out, lo, hi,
                )
                df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi))

    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using time-aware interpolation.

    Strategy per column:
      irradiance    → linear interpolation (short gaps) + forward fill (longer gaps)
      weather cols  → linear interpolation is safe (weather changes gradually)
      Night-time irradiance gaps → fill with 0 (it's dark, not missing)
    """
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        pct_missing = n_missing / len(df) * 100
        logger.info("Column '%s': %.1f%% missing values", col, pct_missing)

        if pct_missing > 50:
            logger.warning(
                "Column '%s' is >50%% missing — results may be unreliable", col
            )

        # Linear interpolation for short gaps (up to 6 hours)
        df[col] = df[col].interpolate(method="time", limit=6)

        # Forward/backward fill for remaining gaps
        df[col] = df[col].ffill().bfill()

    # Zero-fill any remaining NaN in irradiance (nighttime sensor gaps)
    if "irradiance" in df.columns:
        df["irradiance"] = df["irradiance"].fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. GHI estimation (Angstrom-Prescott formula)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_ghi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Global Horizontal Irradiance from weather variables using a
    modified Angstrom-Prescott clearness index model.

    Formula:
      1. Start with base clearness index = 0.75
      2. Adjust for cloud cover:    clearness × (1 - 0.75 × cloud_fraction^3.4)
      3. Adjust for humidity:       clearness × (1 - 0.1 × rh_fraction)
      4. Adjust for sunshine hours: clearness × (0.25 + 0.75 × sun_fraction)
      5. GHI = clearness × solar_constant × daytime_factor

    The daytime_factor uses a sinusoidal model based on hour of day to
    approximate the sun's elevation angle without needing latitude input.

    This is physics-informed estimation, not ML — it's fast, explainable,
    and appropriate when no measured irradiance data is available.
    """
    logger.info("Estimating GHI from weather variables (Angstrom-Prescott model)")

    n     = len(df)
    hours = df.index.hour

    # ── Daytime solar elevation factor ───────────────────────────────────────
    # Approximates sin(solar_elevation) using hour of day
    # Peak at solar noon (12:00), zero at sunrise (~6:00) and sunset (~18:00)
    daytime_mask   = (hours >= 6) & (hours <= 18)
    solar_angle    = np.where(
        daytime_mask,
        np.sin(np.pi * (hours - 6) / 12),   # 0 at 6am, 1 at noon, 0 at 6pm
        0.0,
    )

    # ── Base clearness index ──────────────────────────────────────────────────
    clearness = np.full(n, 0.75)

    # ── Cloud cover adjustment ────────────────────────────────────────────────
    if "cloud_cover" in df.columns:
        cloud_fraction = df["cloud_cover"].values / 100.0
        cloud_fraction = np.clip(cloud_fraction, 0.0, 1.0)
        clearness     *= (1.0 - 0.75 * cloud_fraction ** 3.4)
        logger.info("Cloud cover adjustment applied")

    # ── Humidity adjustment ───────────────────────────────────────────────────
    if "humidity" in df.columns:
        rh_fraction = df["humidity"].values / 100.0
        rh_fraction = np.clip(rh_fraction, 0.0, 1.0)
        clearness  *= (1.0 - 0.1 * rh_fraction)
        logger.info("Humidity adjustment applied")

    # ── Sunshine hours adjustment ─────────────────────────────────────────────
    if "sunshine_hours" in df.columns:
        # Normalise to fraction of maximum possible daylight (12 hours)
        sun_fraction = np.clip(df["sunshine_hours"].values / 12.0, 0.0, 1.0)
        clearness   *= (0.25 + 0.75 * sun_fraction)
        logger.info("Sunshine hours adjustment applied")

    # ── Final GHI calculation ─────────────────────────────────────────────────
    ghi = clearness * SOLAR_CONSTANT * solar_angle
    ghi = np.clip(ghi, IRRADIANCE_MIN, IRRADIANCE_MAX)

    df["irradiance"] = ghi

    estimated_mean = ghi[daytime_mask].mean() if daytime_mask.any() else 0
    logger.info(
        "GHI estimated: mean daytime value = %.1f W/m²", estimated_mean
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public interface — called by the upload router
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    file_bytes:     bytes,
    detected_cols:  dict[str, str | None],
    detection_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Args:
        file_bytes:     Raw bytes from the uploaded CSV file
        detected_cols:  Column mapping from ai_detector.detect_columns()
                        e.g. {"irradiance": "GHI", "temperature": "Temp_C", ...}
        detection_mode: "direct" | "estimated"

    Returns:
        Tuple of:
          - df:   Clean DataFrame with DatetimeIndex and standardised columns
          - meta: Dict with processing statistics for the API response
                  {rows_raw, rows_clean, pct_clean, columns_available,
                   irradiance_mean, irradiance_max, date_range_days}

    Raises:
        ValueError for unrecoverable data issues (empty file, bad timestamps)
    """
    logger.info("Preprocessing pipeline started (mode=%s)", detection_mode)

    # ── Step 1: Parse CSV ────────────────────────────────────────────────────
    df = parse_csv(file_bytes)
    rows_raw = len(df)

    # ── Step 2: Standardise column names ────────────────────────────────────
    df = standardise_columns(df, detected_cols)

    # ── Step 3: Parse timestamps ─────────────────────────────────────────────
    df = parse_timestamps(df)

    # ── Step 4: Coerce to numeric + clip to physical bounds ──────────────────
    df = _coerce_numeric(df)
    df = _clip_to_bounds(df)

    # ── Step 5: Estimate GHI if no irradiance column ─────────────────────────
    if detection_mode == "estimated" and "irradiance" not in df.columns:
        df = _estimate_ghi(df)

    # ── Step 6: Fill missing values ──────────────────────────────────────────
    df = _fill_missing(df)

    # ── Step 7: Final validation ─────────────────────────────────────────────
    if "irradiance" not in df.columns:
        raise ValueError(
            "No irradiance column available after preprocessing. "
            "Upload a CSV with a GHI/irradiance column or at least "
            "one weather variable (temperature, cloud cover, or humidity)."
        )

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} rows remain after cleaning — "
            f"minimum {MIN_ROWS} required for reliable forecasting."
        )

    rows_clean = len(df)

    # ── Step 8: Build metadata summary ───────────────────────────────────────
    date_range_days = (df.index[-1] - df.index[0]).days + 1
    irr             = df["irradiance"]

    meta: dict[str, Any] = {
        "rows_raw":            rows_raw,
        "rows_clean":          rows_clean,
        "pct_clean":           round(rows_clean / rows_raw * 100, 1),
        "columns_available":   list(df.columns),
        "irradiance_mean":     round(float(irr.mean()), 2),
        "irradiance_max":      round(float(irr.max()),  2),
        "irradiance_min":      round(float(irr.min()),  2),
        "date_range_days":     date_range_days,
        "date_start":          str(df.index[0]),
        "date_end":            str(df.index[-1]),
        "detection_mode":      detection_mode,
    }

    logger.info(
        "Preprocessing complete | rows=%d→%d | irradiance mean=%.1f W/m² | span=%d days",
        rows_raw, rows_clean,
        meta["irradiance_mean"],
        date_range_days,
    )

    return df, meta
# Handles CSV parsing, cleaning, and feature engineering preparation.

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Physical limits — anything outside these is a sensor error, not real data
IRRADIANCE_MIN =    0.0
IRRADIANCE_MAX = 1400.0

TEMP_MIN,  TEMP_MAX  = -50.0,  60.0
HUM_MIN,   HUM_MAX   =   0.0, 100.0
WIND_MIN,  WIND_MAX  =   0.0,  75.0
CLOUD_MIN, CLOUD_MAX =   0.0, 100.0
SUN_MIN,   SUN_MAX   =   0.0,  24.0

# Need at least 2 days of hourly data to train reliably
MIN_ROWS = 48

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    # Try multiple encodings and separators; handles NSRDB multi-row headers too.
    encodings  = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1"]
    separators = [",", ";", "\t", "|"]

    def _looks_like_metadata(df: pd.DataFrame) -> bool:
        # True if the first row looks like a metadata header, not column names.
        meta_indicators = ["source", "location id", "version", "units"]
        col_lower = [str(c).lower() for c in df.columns]
        return any(ind in col_lower for ind in meta_indicators)

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep=sep,
                    encoding=encoding,
                    low_memory=False,
                )

                if df.shape[1] < 2:
                    continue

                if _looks_like_metadata(df):
                    logger.info("Detected multi-row header format (e.g. NSRDB) — skipping metadata row")

                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        sep=sep,
                        encoding=encoding,
                        low_memory=False,
                        skiprows=[0, 1],
                    )

                    # NSRDB sometimes has a units row (e.g. "w/m2", "%") right after headers
                    first_row = df.iloc[0].astype(str).str.lower()
                    unit_indicators = ["w/m2", "w/m²", "%", " c", "m/s", "degree", "mbar", "cm", "kwh"]
                    is_units_row = any(
                        any(unit in val for unit in unit_indicators)
                        for val in first_row.values
                    )

                    if is_units_row:
                        logger.info("Units row detected in CSV — skipping it too")
                        df = pd.read_csv(
                            io.BytesIO(file_bytes),
                            sep=sep,
                            encoding=encoding,
                            low_memory=False,
                            skiprows=[0, 2],
                        )

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


def standardise_columns(
    df:      pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    # Rename detected columns to internal standard names and drop everything else.
    rename_map = {
        original: standard
        for standard, original in mapping.items()
        if original is not None
    }

    cols_to_keep = [c for c in df.columns if c in rename_map]
    df = df[cols_to_keep].rename(columns=rename_map)

    logger.info("Standardised columns: %s", list(df.columns))
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    # Convert the timestamp column to a DatetimeIndex using multiple fallback layers.
    col_lower_map = {c.strip().lower(): c for c in df.columns}

    # NSRDB stores date parts in separate Year/Month/Day/Hour columns
    if all(k in col_lower_map for k in ["year", "month", "day", "hour"]):
        logger.info("Detected NSRDB-style date columns — combining Year/Month/Day/Hour")
        try:
            minute_col = col_lower_map.get("minute")
            build = dict(
                year  = pd.to_numeric(df[col_lower_map["year"]],  errors="coerce"),
                month = pd.to_numeric(df[col_lower_map["month"]], errors="coerce"),
                day   = pd.to_numeric(df[col_lower_map["day"]],   errors="coerce"),
                hour  = pd.to_numeric(df[col_lower_map["hour"]],  errors="coerce"),
            )
            if minute_col:
                build["minute"] = pd.to_numeric(df[minute_col], errors="coerce")

            df["timestamp"] = pd.to_datetime(build)

            drop_cols = [col_lower_map[k] for k in ["year", "month", "day", "hour"] if k in col_lower_map]
            if minute_col:
                drop_cols.append(minute_col)
            df = df.drop(columns=drop_cols, errors="ignore")

            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df[df.index.notna()]

            logger.info("Timestamp index: %s → %s (%d rows)", df.index[0], df.index[-1], len(df))
            return df

        except Exception as exc:
            logger.warning("NSRDB date combine failed: %s — falling back", exc)

    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found — generating synthetic hourly index")
        df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq="h")
        df = df.set_index("timestamp").sort_index()
        return df

    parsed = None

    # Layer 1: let pandas figure out the format automatically
    try:
        parsed = pd.to_datetime(df["timestamp"])
        if parsed.isna().sum() / len(parsed) < 0.5:
            logger.info("Timestamps parsed via pandas auto-detect")
    except Exception:
        parsed = None

    # Layer 2: try common explicit formats one by one
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

    # Layer 3: dateutil fuzzy parser handles almost any human-readable format
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

    # Layer 4: treat as Unix timestamp (seconds or milliseconds)
    if parsed is None or parsed.isna().all():
        try:
            numeric = pd.to_numeric(df["timestamp"], errors="coerce")
            if numeric.notna().sum() > len(df) * 0.5:
                if numeric.median() > 1e10:
                    parsed = pd.to_datetime(numeric, unit="ms")
                else:
                    parsed = pd.to_datetime(numeric, unit="s")
                logger.info("Timestamps parsed as Unix timestamp")
        except Exception:
            parsed = None

    # Layer 5: nothing worked — generate a synthetic index so we never crash
    if parsed is None or parsed.isna().all():
        logger.warning("Could not parse timestamps — generating synthetic hourly index")
        parsed = pd.date_range(start="2020-01-01", periods=len(df), freq="h")

    df["timestamp"] = parsed

    df = df.set_index("timestamp").sort_index()

    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Removed %d duplicate timestamps", n_dupes)
        df = df[~df.index.duplicated(keep="last")]

    logger.info("Timestamp index: %s → %s (%d rows)", df.index[0], df.index[-1], len(df))
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Force all columns to float; non-numeric strings become NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clip_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    # Out-of-range values are sensor errors, so replace them with NaN rather than clamping.
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
    # Interpolate short gaps, forward/back fill the rest, zero-fill night irradiance.
    df = df[df.index.notna()]
    if len(df) == 0:
        return df

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        pct_missing = n_missing / len(df) * 100
        logger.info("Column '%s': %.1f%% missing values", col, pct_missing)

        if pct_missing > 50:
            logger.warning("Column '%s' is >50%% missing — results may be unreliable", col)

        df[col] = df[col].interpolate(method="time", limit=6)
        df[col] = df[col].ffill().bfill()

    # Nighttime irradiance gaps are genuinely zero, not missing
    if "irradiance" in df.columns:
        df["irradiance"] = df["irradiance"].fillna(0.0)

    return df


def preprocess(
    file_bytes:     bytes,
    detected_cols:  dict[str, str | None],
    detection_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full preprocessing pipeline — parse, clean, and return a model-ready DataFrame.

    Args:
        file_bytes:    Raw bytes from the uploaded CSV
        detected_cols: Column mapping from ai_detector.detect_columns()
        detection_mode: Always "direct" — irradiance column must be present in the CSV

    Returns:
        (df, meta) — cleaned DataFrame with DatetimeIndex, plus processing stats

    Raises:
        ValueError if the data is unrecoverable (empty, bad timestamps, etc.)
    """
    logger.info("Preprocessing pipeline started")

    # Step 1: parse the raw CSV bytes
    df = parse_csv(file_bytes)
    rows_raw = len(df)

    # Step 2: build NSRDB timestamp before standardise_columns drops the date columns
    col_lower_map = {c.strip().lower(): c for c in df.columns}
    if all(k in col_lower_map for k in ["year", "month", "day", "hour"]):
        logger.info("NSRDB date columns detected — building timestamp before standardising")
        try:
            minute_col = col_lower_map.get("minute")
            build = dict(
                year  = pd.to_numeric(df[col_lower_map["year"]],  errors="coerce"),
                month = pd.to_numeric(df[col_lower_map["month"]], errors="coerce"),
                day   = pd.to_numeric(df[col_lower_map["day"]],   errors="coerce"),
                hour  = pd.to_numeric(df[col_lower_map["hour"]],  errors="coerce"),
            )
            if minute_col:
                build["minute"] = pd.to_numeric(df[minute_col], errors="coerce")

            df["timestamp"] = pd.to_datetime(build, errors="coerce")

            drop_cols = [col_lower_map[k] for k in ["year", "month", "day", "hour"] if k in col_lower_map]
            if minute_col:
                drop_cols.append(minute_col)
            df = df.drop(columns=drop_cols, errors="ignore")

            detected_cols = {**detected_cols, "timestamp": "timestamp"}
            logger.info("NSRDB timestamp built successfully")
        except Exception as exc:
            logger.warning("NSRDB timestamp build failed: %s", exc)

    # Step 3: rename columns to standard names
    df = standardise_columns(df, detected_cols)

    # Step 4: parse timestamps into a DatetimeIndex
    df = parse_timestamps(df)

    # Step 5: drop any rows where the timestamp couldn't be parsed
    nat_count = df.index.isna().sum()
    if nat_count > 0:
        logger.warning("Dropping %d rows with NaT timestamps", nat_count)
        df = df[df.index.notna()]

    # Step 6: coerce to float and remove physically impossible values
    df = _coerce_numeric(df)
    df = _clip_to_bounds(df)

    # Step 7: validate irradiance BEFORE filling — once filled NaNs become 0 and are invisible
    if "irradiance" not in df.columns:
        raise ValueError(
            "No irradiance column found. Please upload a CSV that contains a GHI or irradiance column."
        )

    irr_missing = df["irradiance"].isna().sum()
    irr_pct_missing = irr_missing / len(df) * 100
    irr_nonzero = (df["irradiance"] > 0).sum()

    if irr_nonzero == 0 and irr_missing == len(df):
        raise ValueError(
            "The GHI/irradiance column is entirely empty. "
            "Please check your CSV — the irradiance values appear to be missing or not mapped correctly."
        )

    if irr_pct_missing > 80:
        raise ValueError(
            f"The GHI/irradiance column is {irr_pct_missing:.0f}% empty. "
            "Too little data to produce a reliable forecast — please check your CSV."
        )

    # Step 8: fill remaining gaps
    df = _fill_missing(df)

    # Step 9: make sure we ended up with enough rows
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} rows remain after cleaning — "
            f"minimum {MIN_ROWS} required for reliable forecasting."
        )

    rows_clean = len(df)

    # Step 9: build a summary for the API response
    date_range_days = (df.index[-1] - df.index[0]).days + 1
    irr             = df["irradiance"]
    hourly_avg = (
        irr.groupby(df.index.hour)
        .mean()
        .reindex(range(24), fill_value=0.0)
        .round(2)
        .tolist()
    )
    weekday_avg = (
        irr.groupby(df.index.dayofweek)
        .mean()
        .reindex(range(7), fill_value=0.0)
        .round(2)
        .tolist()
    )
    monthly_avg = (
        irr.groupby(df.index.month)
        .mean()
        .reindex(range(1, 13), fill_value=0.0)
        .round(2)
        .tolist()
    )
    daily_avg_df = irr.resample("D").mean().round(2).reset_index()
    date_col = "timestamp" if "timestamp" in daily_avg_df.columns else daily_avg_df.columns[0]
    daily_avg = [
        {
            "date": str(row[date_col])[:10],
            "avg": float(row["irradiance"]),
        }
        for _, row in daily_avg_df.iterrows()
    ]

    meta: dict[str, Any] = {
        "rows_raw":          rows_raw,
        "rows_clean":        rows_clean,
        "pct_clean":         round(rows_clean / rows_raw * 100, 1),
        "columns_available": list(df.columns),
        "irradiance_mean":   round(float(irr.mean()), 2),
        "irradiance_max":    round(float(irr.max()),  2),
        "irradiance_min":    round(float(irr.min()),  2),
        "date_range_days":   date_range_days,
        "date_start":        str(df.index[0]),
        "date_end":          str(df.index[-1]),
        "hourly_avg":         hourly_avg,
        "weekday_avg":        weekday_avg,
        "monthly_avg":        monthly_avg,
        "daily_avg":          daily_avg,
    }

    logger.info(
        "Preprocessing complete | rows=%d→%d | irradiance mean=%.1f W/m² | span=%d days",
        rows_raw, rows_clean,
        meta["irradiance_mean"],
        date_range_days,
    )

    return df, meta

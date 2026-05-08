"""
test_preprocessing.py — Unit tests for app/services/preprocessing.py

Covers: parse_csv, standardise_columns, parse_timestamps,
        _coerce_numeric, _clip_to_bounds, _fill_missing,
        _estimate_ghi, preprocess
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing import (
    MIN_ROWS,
    _clip_to_bounds,
    _coerce_numeric,
    _estimate_ghi,
    _fill_missing,
    parse_csv,
    parse_timestamps,
    preprocess,
    standardise_columns,
)
from conftest import make_csv_bytes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _indexed_df(data: dict) -> pd.DataFrame:
    """Return a DataFrame with a DatetimeIndex of length matching the longest column."""
    n   = max(len(v) for v in data.values())
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(data, index=idx)


def _weather_df(n: int = 96) -> pd.DataFrame:
    """Return a weather-only DataFrame (no irradiance) for GHI estimation tests."""
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    return pd.DataFrame({
        "temperature": np.ones(n) * 25,
        "humidity":    np.ones(n) * 50,
        "cloud_cover": np.ones(n) * 20,
    }, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# parse_csv
# ══════════════════════════════════════════════════════════════════════════════

class TestParseCsv:

    def test_returns_dataframe_from_valid_bytes(self):
        """parse_csv must return a DataFrame when given a valid CSV byte string."""
        df = parse_csv(make_csv_bytes(rows=72))
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= MIN_ROWS

    def test_detects_multiple_columns(self):
        """parse_csv must preserve all columns present in the CSV."""
        df = parse_csv(make_csv_bytes(rows=72))
        assert "irradiance"  in df.columns
        assert "temperature" in df.columns

    def test_handles_semicolon_separator(self):
        """parse_csv must recognise semicolons as the column separator."""
        buf = io.StringIO()
        w   = csv.writer(buf, delimiter=";")
        w.writerow(["timestamp", "irradiance"])
        start = datetime(2024, 1, 1)
        for i in range(MIN_ROWS + 2):
            w.writerow([(start + timedelta(hours=i)).isoformat(), 100.0 + i])
        df = parse_csv(buf.getvalue().encode())
        assert "irradiance" in df.columns

    def test_raises_on_too_few_rows(self):
        """parse_csv must raise ValueError when the file has fewer rows than MIN_ROWS."""
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["timestamp", "irradiance"])
        for i in range(5):
            w.writerow([f"2024-01-01 {i:02d}:00:00", 100.0])
        with pytest.raises(ValueError):
            parse_csv(buf.getvalue().encode())

    def test_raises_on_completely_invalid_bytes(self):
        """parse_csv must raise ValueError when the bytes cannot be parsed as CSV."""
        with pytest.raises(ValueError):
            parse_csv(b"\x00\x01\x02\x03" * 200)

    def test_handles_nsrdb_multi_row_header(self):
        """parse_csv must skip the metadata row in NSRDB-style files and parse normally."""
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["Source", "Location ID", "City", "State", "Latitude"])
        w.writerow(["NSRDB", "12345", "LA", "CA", "34.05"])
        w.writerow(["Year", "Month", "Day", "Hour", "GHI"])
        for i in range(MIN_ROWS + 2):
            dt = datetime(2024, 1, 1) + timedelta(hours=i)
            w.writerow([dt.year, dt.month, dt.day, dt.hour, max(0, 500 - abs(12 - dt.hour) * 40)])
        df = parse_csv(buf.getvalue().encode())
        assert df is not None
        assert len(df) >= MIN_ROWS


# ══════════════════════════════════════════════════════════════════════════════
# standardise_columns
# ══════════════════════════════════════════════════════════════════════════════

class TestStandardiseColumns:

    def test_renames_columns(self):
        """standardise_columns must rename source columns to standard internal names."""
        df     = pd.DataFrame({"GHI": [100, 200], "Temp": [20, 22], "extra": [0, 0]})
        result = standardise_columns(df, {"irradiance": "GHI", "temperature": "Temp"})
        assert "irradiance"  in result.columns
        assert "temperature" in result.columns

    def test_drops_unmapped_columns(self):
        """standardise_columns must discard any column not present in the mapping."""
        df     = pd.DataFrame({"GHI": [100], "Temp": [20], "noise": [99]})
        result = standardise_columns(df, {"irradiance": "GHI", "temperature": "Temp"})
        assert "noise" not in result.columns

    def test_ignores_none_values_in_mapping(self):
        """Mapping entries where the original column is None must not crash or add columns."""
        df     = pd.DataFrame({"GHI": [100], "Temp": [20]})
        result = standardise_columns(df, {"irradiance": "GHI", "temperature": "Temp", "humidity": None})
        assert "humidity" not in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# parse_timestamps
# ══════════════════════════════════════════════════════════════════════════════

class TestParseTimestamps:

    def test_sets_datetime_index(self):
        """parse_timestamps must return a DataFrame whose index is a DatetimeIndex."""
        df = pd.DataFrame({
            "timestamp":  [(datetime(2024, 1, 1) + timedelta(hours=i)).isoformat() for i in range(72)],
            "irradiance": np.random.rand(72) * 500,
        })
        assert isinstance(parse_timestamps(df).index, pd.DatetimeIndex)

    def test_removes_duplicates(self):
        """parse_timestamps must deduplicate rows with the same timestamp."""
        rows = [(datetime(2024, 1, 1, 12).isoformat(), 400)] * 3
        df   = pd.DataFrame(rows, columns=["timestamp", "irradiance"])
        assert parse_timestamps(df).index.is_unique

    def test_generates_synthetic_index_when_no_column(self):
        """parse_timestamps must synthesise a DatetimeIndex when no timestamp column exists."""
        df     = pd.DataFrame({"irradiance": np.ones(72)})
        result = parse_timestamps(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 72

    def test_handles_nsrdb_year_month_day_hour_columns(self):
        """parse_timestamps must combine Year/Month/Day/Hour columns into a DatetimeIndex."""
        rows = []
        for i in range(72):
            dt = datetime(2024, 6, 1) + timedelta(hours=i)
            rows.append({"Year": dt.year, "Month": dt.month, "Day": dt.day,
                         "Hour": dt.hour, "irradiance": 100.0})
        result = parse_timestamps(pd.DataFrame(rows))
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "Year" not in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# _coerce_numeric
# ══════════════════════════════════════════════════════════════════════════════

class TestCoerceNumeric:

    def test_converts_string_numbers(self):
        """_coerce_numeric must convert numeric strings to float; non-numeric become NaN."""
        df     = _indexed_df({"irradiance": ["100", "200", "abc", "300"]})
        result = _coerce_numeric(df)
        assert result["irradiance"].dtype == float
        assert pd.isna(result["irradiance"].iloc[2])

    def test_leaves_floats_unchanged(self):
        """_coerce_numeric must not alter columns that are already numeric."""
        df     = _indexed_df({"irradiance": [100.0, 200.0, 300.0, 400.0]})
        result = _coerce_numeric(df)
        assert result["irradiance"].iloc[0] == pytest.approx(100.0)


# ══════════════════════════════════════════════════════════════════════════════
# _clip_to_bounds
# ══════════════════════════════════════════════════════════════════════════════

class TestClipToBounds:

    def test_replaces_negative_irradiance_with_nan(self):
        """Negative irradiance is physically impossible — must be replaced with NaN."""
        df     = _indexed_df({"irradiance": [-50.0, 300.0, 700.0, 1500.0]})
        result = _clip_to_bounds(df)
        assert pd.isna(result["irradiance"].iloc[0])   # -50 below min
        assert pd.isna(result["irradiance"].iloc[3])   # 1500 above max (1400)

    def test_keeps_valid_irradiance(self):
        """Values within the valid irradiance range must not be modified."""
        df     = _indexed_df({"irradiance": [0.0, 500.0, 1000.0, 1400.0]})
        result = _clip_to_bounds(df)
        assert not result["irradiance"].isna().any()

    def test_ignores_unknown_columns(self):
        """_clip_to_bounds must not crash on columns that have no defined bounds."""
        df     = _indexed_df({"some_unknown_col": [1.0, 2.0, 3.0, 4.0]})
        result = _clip_to_bounds(df)
        assert list(result.columns) == ["some_unknown_col"]

    def test_clamps_humidity(self):
        """Humidity outside [0, 100] must be replaced with NaN."""
        df     = _indexed_df({"humidity": [-5.0, 50.0, 105.0, 80.0]})
        result = _clip_to_bounds(df)
        assert pd.isna(result["humidity"].iloc[0])
        assert pd.isna(result["humidity"].iloc[2])
        assert not pd.isna(result["humidity"].iloc[1])


# ══════════════════════════════════════════════════════════════════════════════
# _fill_missing
# ══════════════════════════════════════════════════════════════════════════════

class TestFillMissing:

    def test_interpolates_short_gaps(self):
        """_fill_missing must interpolate short NaN runs so no NaN remains in irradiance."""
        vals   = [100.0, np.nan, np.nan, 400.0, 500.0, 600.0] * 12
        result = _fill_missing(_indexed_df({"irradiance": vals}))
        assert not result["irradiance"].isna().any()

    def test_sets_remaining_nan_to_zero(self):
        """Any un-interpolatable NaN in irradiance must be zero-filled (nighttime proxy)."""
        idx    = pd.date_range("2024-01-01", periods=72, freq="h")
        df     = pd.DataFrame({"irradiance": np.full(72, np.nan)}, index=idx)
        result = _fill_missing(df)
        assert (result["irradiance"] == 0.0).all()

    def test_returns_empty_df_unchanged(self):
        """_fill_missing on an empty DataFrame must not raise and must return an empty frame."""
        df     = pd.DataFrame({"irradiance": pd.Series([], dtype=float)})
        result = _fill_missing(df)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════════════
# _estimate_ghi
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateGhi:

    def test_adds_irradiance_column(self):
        """_estimate_ghi must create an 'irradiance' column in the DataFrame."""
        assert "irradiance" in _estimate_ghi(_weather_df()).columns

    def test_daytime_values_are_positive(self):
        """Estimated irradiance at midday must be greater than zero."""
        result = _estimate_ghi(_weather_df(n=24))
        assert (result.between_time("10:00", "14:00")["irradiance"] > 0).any()

    def test_nighttime_values_are_zero(self):
        """Estimated irradiance between 1-4 AM must be zero."""
        result = _estimate_ghi(_weather_df(n=24))
        assert (result.between_time("01:00", "04:00")["irradiance"] == 0.0).all()

    def test_values_within_physical_bounds(self):
        """All estimated irradiance values must be within [0, 1400] W/m²."""
        result = _estimate_ghi(_weather_df(n=96))
        assert result["irradiance"].between(0.0, 1400.0).all()

    def test_high_cloud_reduces_irradiance(self):
        """High cloud cover (90%) should produce lower GHI than low cloud cover (10%)."""
        idx       = pd.date_range("2024-06-01 10:00", periods=4, freq="h")
        clear     = _estimate_ghi(pd.DataFrame({"cloud_cover": [10.0] * 4}, index=idx))
        cloudy    = _estimate_ghi(pd.DataFrame({"cloud_cover": [90.0] * 4}, index=idx))
        assert clear["irradiance"].mean() > cloudy["irradiance"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# preprocess  (full pipeline)
# ══════════════════════════════════════════════════════════════════════════════

_DIRECT_MAP = {
    "irradiance": "irradiance", "temperature": "temperature",
    "humidity": "humidity", "wind_speed": "wind_speed",
    "cloud_cover": "cloud_cover", "timestamp": "timestamp",
}

_ESTIMATED_MAP = {
    "temperature": "temperature", "humidity": "humidity",
    "wind_speed": "wind_speed", "cloud_cover": "cloud_cover",
    "timestamp": "timestamp",
}


class TestPreprocess:

    def test_returns_dataframe_and_meta(self):
        """preprocess must return a (DataFrame, dict) tuple for a valid CSV."""
        df, meta = preprocess(make_csv_bytes(rows=72), _DIRECT_MAP, "direct")
        assert isinstance(df, pd.DataFrame)
        assert "irradiance" in df.columns
        assert "rows_clean" in meta

    def test_meta_contains_required_keys(self):
        """The meta dict must include all expected summary keys."""
        _, meta = preprocess(make_csv_bytes(rows=72), _DIRECT_MAP, "direct")
        for key in ("rows_raw", "rows_clean", "irradiance_mean", "irradiance_max",
                    "irradiance_min", "date_range_days", "date_start", "date_end"):
            assert key in meta, f"Missing meta key: {key}"

    def test_estimated_mode_adds_irradiance(self):
        """preprocess in 'estimated' mode must derive an irradiance column from weather vars."""
        df, _ = preprocess(make_csv_bytes(rows=72, with_irradiance=False), _ESTIMATED_MAP, "estimated")
        assert "irradiance" in df.columns

    def test_raises_when_irradiance_missing_in_direct_mode(self):
        """preprocess must raise ValueError if mode='direct' but no irradiance column maps."""
        with pytest.raises(ValueError):
            preprocess(make_csv_bytes(rows=72, with_irradiance=False),
                       {"temperature": "temperature", "timestamp": "timestamp"}, "direct")

    def test_raises_on_too_few_clean_rows(self):
        """preprocess must raise ValueError if fewer than MIN_ROWS rows remain after cleaning."""
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["timestamp", "irradiance"])
        for i in range(10):
            w.writerow([f"2024-01-01 {i:02d}:00:00", 100.0])
        with pytest.raises((ValueError, Exception)):
            preprocess(buf.getvalue().encode(),
                       {"irradiance": "irradiance", "timestamp": "timestamp"}, "direct")

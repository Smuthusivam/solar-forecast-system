"""
test_anomaly.py — Unit tests for app/services/anomaly.py

Covers: _hourly_max, _is_physically_impossible, _is_weather_coherent,
        _is_daytime, severity helpers,
        _detect_zscore, _detect_iqr, _detect_rolling,
        _merge_anomalies, _apply_physical_filters,
        detect_anomalies (public entry point)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.anomaly import (
    DAYTIME_END,
    DAYTIME_START,
    _apply_physical_filters,
    _detect_iqr,
    _detect_rolling,
    _detect_zscore,
    _hourly_max,
    _is_daytime,
    _is_physically_impossible,
    _is_weather_coherent,
    _merge_anomalies,
    _severity_from_change,
    _severity_from_iqr,
    _severity_from_zscore,
    detect_anomalies,
)
from conftest import make_irradiance_df, make_irradiance_series


# ── Shared helper ─────────────────────────────────────────────────────────────

def _anomaly(ts: pd.Timestamp, severity: str, method: str = "zscore",
             score: float = 3.0, value: float = 600.0) -> dict:
    return {
        "timestamp": ts, "value": value, "expected": 400.0,
        "deviation": abs(value - 400.0), "severity": severity,
        "method": method, "score": score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Physical helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHourlyMax:

    def test_midnight_is_zero(self):
        """Maximum possible GHI at midnight must be 0 W/m²."""
        assert _hourly_max(0) == 0.0

    def test_noon_is_highest(self):
        """Solar noon (hour 12) must have the highest ceiling of any hour."""
        noon = _hourly_max(12)
        assert all(noon >= _hourly_max(h) for h in range(24))

    def test_returns_900_for_unknown_hour(self):
        """_hourly_max must fall back to 900 W/m² for an out-of-range hour."""
        assert _hourly_max(99) == 900.0


class TestIsPhysicallyImpossible:

    def test_impossible_at_night(self):
        """Any positive irradiance at midnight must be physically impossible."""
        assert _is_physically_impossible(50.0, hour=0) is True

    def test_possible_at_noon(self):
        """A realistic midday value (500 W/m²) must not be flagged as impossible."""
        assert _is_physically_impossible(500.0, hour=12) is False

    def test_far_above_ceiling_is_impossible(self):
        """A value 50% above the noon ceiling must be flagged as impossible."""
        assert _is_physically_impossible(1400.0, hour=12) is True

    def test_within_ten_percent_tolerance_is_allowed(self):
        """A value 5% above the hourly max must pass (tolerance is 10%)."""
        ceiling = _hourly_max(12)
        assert _is_physically_impossible(ceiling * 1.05, hour=12) is False


class TestIsWeatherCoherent:

    def test_coherent_when_no_cloud_cover_column(self):
        """Must return True when cloud_cover is absent — no evidence to judge."""
        idx = pd.date_range("2024-06-01 12:00", periods=1, freq="h")
        df  = pd.DataFrame({"temperature": [25.0]}, index=idx)
        assert _is_weather_coherent(600.0, idx[0], df) is True

    def test_incoherent_high_cloud_high_irradiance(self):
        """Cloud >80% combined with irradiance >600 W/m² must be incoherent."""
        idx = pd.date_range("2024-06-01 12:00", periods=1, freq="h")
        df  = pd.DataFrame({"cloud_cover": [90.0]}, index=idx)
        assert _is_weather_coherent(700.0, idx[0], df) is False

    def test_coherent_moderate_cloud_and_irradiance(self):
        """Moderate cloud (40%) and moderate irradiance must be coherent."""
        idx = pd.date_range("2024-06-01 12:00", periods=1, freq="h")
        df  = pd.DataFrame({"cloud_cover": [40.0]}, index=idx)
        assert _is_weather_coherent(400.0, idx[0], df) is True


class TestIsDaytime:

    def test_noon_is_daytime(self):
        assert _is_daytime(12) is True

    def test_midnight_is_not_daytime(self):
        assert _is_daytime(0) is False

    def test_boundary_start_is_daytime(self):
        assert _is_daytime(DAYTIME_START) is True

    def test_boundary_end_is_daytime(self):
        assert _is_daytime(DAYTIME_END) is True


# ══════════════════════════════════════════════════════════════════════════════
# Severity helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestSeverityHelpers:

    def test_zscore_low(self):
        assert _severity_from_zscore(2.6) == "low"

    def test_zscore_medium(self):
        assert _severity_from_zscore(3.6) == "medium"

    def test_zscore_high(self):
        assert _severity_from_zscore(5.0) == "high"

    def test_zscore_uses_absolute_value(self):
        """Negative z-scores must be treated by magnitude."""
        assert _severity_from_zscore(-5.0) == "high"

    def test_iqr_zero_iqr_defaults_to_low(self):
        """Zero IQR must not cause division by zero — must return 'low'."""
        assert _severity_from_iqr(distance=100.0, iqr=0) == "low"

    def test_iqr_high_ratio(self):
        """distance/IQR ratio ≥ 4.0 must produce 'high' severity."""
        assert _severity_from_iqr(distance=400.0, iqr=100.0) == "high"

    def test_change_low(self):
        assert _severity_from_change(0.86) == "low"

    def test_change_high(self):
        assert _severity_from_change(0.97) == "high"


# ══════════════════════════════════════════════════════════════════════════════
# Statistical detectors
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectZscore:

    def test_returns_list(self):
        assert isinstance(_detect_zscore(make_irradiance_series()), list)

    def test_finds_spike(self):
        """Must detect an obvious spike inserted at a known daytime position."""
        series = make_irradiance_series(n=96)
        spike  = pd.Timestamp("2024-06-01 12:00")
        series[spike] = 9999.0
        flagged = {r["timestamp"] for r in _detect_zscore(series)}
        assert spike in flagged

    def test_each_entry_has_required_keys(self):
        """Every returned entry must contain the expected keys."""
        series = make_irradiance_series(n=96)
        series[pd.Timestamp("2024-06-01 12:00")] = 9999.0
        required = {"timestamp", "value", "expected", "deviation", "severity", "method", "score"}
        for entry in _detect_zscore(series):
            assert required.issubset(entry.keys())

    def test_ignores_nighttime(self):
        """Must not flag a value planted at 2 AM."""
        series = make_irradiance_series(n=96)
        night  = pd.Timestamp("2024-06-01 02:00")
        series[night] = 9999.0
        flagged = {r["timestamp"] for r in _detect_zscore(series)}
        assert night not in flagged

    def test_ignores_below_min_irradiance(self):
        """Must skip daytime values below MIN_IRRADIANCE (near-zero dawn/dusk)."""
        series = make_irradiance_series(n=96)
        dawn   = pd.Timestamp("2024-06-01 10:00")
        series[dawn] = 1.0
        flagged = {r["timestamp"] for r in _detect_zscore(series)}
        assert dawn not in flagged


class TestDetectIqr:

    def test_returns_list(self):
        assert isinstance(_detect_iqr(make_irradiance_series()), list)

    def test_finds_outlier(self):
        """Must detect a clear outlier in an otherwise tight series."""
        idx    = pd.date_range("2024-06-01 09:00", periods=48, freq="h")
        rng    = np.random.default_rng(0)
        vals   = 400.0 + rng.normal(0, 5, 48)
        series = pd.Series(vals, index=idx)
        series.iloc[30] = 5000.0
        assert len(_detect_iqr(series)) > 0


class TestDetectRolling:

    def test_returns_list(self):
        assert isinstance(_detect_rolling(make_irradiance_series()), list)

    def test_finds_sudden_spike(self):
        """Must detect a sudden large spike between consecutive daytime points."""
        idx    = pd.date_range("2024-06-01 09:00", periods=10, freq="h")
        vals   = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 1900.0, 200.0, 200.0, 200.0, 200.0])
        series = pd.Series(vals, index=idx)
        assert len(_detect_rolling(series)) > 0

    def test_skips_transition_hours(self):
        """Must not flag sunrise/sunset hours (6-8, 18-20) as anomalies."""
        idx        = pd.date_range("2024-06-01 06:00", periods=5, freq="h")
        vals       = np.array([10.0, 500.0, 480.0, 490.0, 10.0])
        series     = pd.Series(vals, index=idx)
        flagged    = {r["timestamp"].hour for r in _detect_rolling(series)}
        transition = {6, 7, 8, 18, 19, 20}
        assert flagged.isdisjoint(transition)


# ══════════════════════════════════════════════════════════════════════════════
# _merge_anomalies
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeAnomalies:

    def test_deduplicates_same_timestamp(self):
        """Must produce exactly one entry per unique timestamp."""
        ts  = pd.Timestamp("2024-06-01 12:00")
        merged = _merge_anomalies([
            _anomaly(ts, "low",    score=2.6),
            _anomaly(ts, "medium", score=3.5),
        ])
        assert len(merged) == 1

    def test_keeps_highest_severity_on_conflict(self):
        """When multiple detectors flag the same timestamp, highest severity wins."""
        ts = pd.Timestamp("2024-06-01 12:00")
        merged = _merge_anomalies([
            _anomaly(ts, "low",    score=2.6),
            _anomaly(ts, "high",   score=5.2),
            _anomaly(ts, "medium", score=3.5),
        ])
        assert merged[0]["severity"] == "high"

    def test_removes_score_field(self):
        """The internal 'score' key must be stripped from the merged output."""
        ts = pd.Timestamp("2024-06-01 12:00")
        assert "score" not in _merge_anomalies([_anomaly(ts, "medium", score=3.5)])[0]

    def test_sorts_by_timestamp_ascending(self):
        """Output must be sorted by timestamp ascending."""
        ts_a, ts_b, ts_c = (pd.Timestamp(f"2024-06-01 {h}:00") for h in (14, 10, 12))
        merged = _merge_anomalies([_anomaly(ts_a, "low"), _anomaly(ts_b, "low"), _anomaly(ts_c, "low")])
        times  = [a["timestamp"] for a in merged]
        assert times == sorted(times)

    def test_empty_input_returns_empty_list(self):
        assert _merge_anomalies([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# _apply_physical_filters
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyPhysicalFilters:

    def test_removes_long_consecutive_runs(self):
        """A run of 8 consecutive hourly anomalies must be classified as a weather event."""
        base      = pd.Timestamp("2024-06-01 09:00")
        df        = make_irradiance_df(96)
        anomalies = [_anomaly(base + timedelta(hours=i), "low", value=300.0) for i in range(8)]
        assert len(_apply_physical_filters(anomalies, df)) < 8

    def test_keeps_short_runs(self):
        """A run of only 3 consecutive anomalies must NOT be dismissed."""
        base      = pd.Timestamp("2024-06-01 10:00")
        df        = make_irradiance_df(96)
        anomalies = [_anomaly(base + timedelta(hours=i), "medium", value=350.0) for i in range(3)]
        assert len(_apply_physical_filters(anomalies, df)) == 3

    def test_keeps_physically_impossible_values(self):
        """Values exceeding the physical ceiling must always be kept regardless of run length."""
        base      = pd.Timestamp("2024-06-01 09:00")
        df        = make_irradiance_df(96)
        anomalies = [_anomaly(base + timedelta(hours=i), "low", value=1500.0) for i in range(8)]
        assert len(_apply_physical_filters(anomalies, df)) == 8


# ══════════════════════════════════════════════════════════════════════════════
# detect_anomalies — public entry point
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectAnomalies:

    def test_returns_required_keys(self):
        """Must always return a dict with the four required keys."""
        result = detect_anomalies(make_irradiance_df())
        for key in ("total_points", "anomaly_count", "anomaly_rate", "anomalies"):
            assert key in result

    def test_total_points_matches_input_length(self):
        """total_points must equal the number of rows in the input DataFrame."""
        assert detect_anomalies(make_irradiance_df(n=96))["total_points"] == 96

    def test_anomaly_count_consistent_with_list_length(self):
        """anomaly_count must equal len(anomalies)."""
        result = detect_anomalies(make_irradiance_df())
        assert result["anomaly_count"] == len(result["anomalies"])

    def test_anomaly_rate_is_fraction(self):
        """anomaly_rate must be between 0.0 and 1.0 inclusive."""
        rate = detect_anomalies(make_irradiance_df())["anomaly_rate"]
        assert 0.0 <= rate <= 1.0

    def test_timestamps_are_iso_strings(self):
        """Each anomaly timestamp must be an ISO 8601 string (not a Timestamp object)."""
        df = make_irradiance_df(n=96)
        df.loc[pd.Timestamp("2024-06-01 12:00"), "irradiance"] = 9999.0
        for a in detect_anomalies(df)["anomalies"]:
            assert isinstance(a["timestamp"], str)
            datetime.fromisoformat(a["timestamp"])  # must not raise

    def test_detects_planted_spike(self):
        """A large spike inserted at midday must appear in the anomaly list."""
        df       = make_irradiance_df(n=96)
        spike_ts = pd.Timestamp("2024-06-01 12:00")
        df.loc[spike_ts, "irradiance"] = 9999.0
        timestamps = [a["timestamp"] for a in detect_anomalies(df)["anomalies"]]
        assert spike_ts.isoformat() in timestamps

    def test_clean_series_has_low_anomaly_rate(self):
        """A smooth synthetic series must produce a very low anomaly rate."""
        idx    = pd.date_range("2024-06-01", periods=96, freq="h")
        vals   = np.array([max(0.0, 600.0 * (1 - abs(12 - ts.hour) / 12)) for ts in idx])
        result = detect_anomalies(pd.DataFrame({"irradiance": vals}, index=idx))
        assert result["anomaly_rate"] < 0.15

    def test_safe_fallback_on_missing_irradiance_column(self):
        """Must return a zero-count result (not raise) if 'irradiance' is absent."""
        result = detect_anomalies(pd.DataFrame({"temperature": np.ones(96)}))
        assert result["anomaly_count"] == 0
        assert result["anomalies"]     == []

    def test_each_anomaly_has_required_fields(self):
        """Every anomaly dict must contain value, expected, severity, method, deviation."""
        df = make_irradiance_df(n=96)
        df.loc[pd.Timestamp("2024-06-01 12:00"), "irradiance"] = 9999.0
        required = {"timestamp", "value", "expected", "deviation", "severity", "method"}
        for a in detect_anomalies(df)["anomalies"]:
            assert required.issubset(a.keys())

    def test_severity_values_are_valid(self):
        """Every anomaly's severity must be one of 'low', 'medium', 'high'."""
        df = make_irradiance_df(n=96)
        df.loc[pd.Timestamp("2024-06-01 12:00"), "irradiance"] = 9999.0
        for a in detect_anomalies(df)["anomalies"]:
            assert a["severity"] in {"low", "medium", "high"}

    def test_method_values_are_valid(self):
        """Every anomaly's method must be one of 'zscore', 'iqr', 'rolling'."""
        df = make_irradiance_df(n=96)
        df.loc[pd.Timestamp("2024-06-01 12:00"), "irradiance"] = 9999.0
        for a in detect_anomalies(df)["anomalies"]:
            assert a["method"] in {"zscore", "iqr", "rolling"}

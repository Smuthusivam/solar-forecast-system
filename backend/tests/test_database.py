"""
test_database.py — Unit tests for app/database.py

Covers: save_forecast_run, get_run_by_id, get_all_runs,
        save_forecast_points, get_forecast_points_by_run_id,
        save_dataset, ForecastRun.__repr__
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.crud import (
    get_all_runs,
    get_forecast_points_by_run_id,
    get_run_by_id,
    save_dataset,
    save_forecast_points,
    save_forecast_run,
)


# Shared kwargs for creating a minimal valid ForecastRun
_RUN = dict(
    filename="test.csv", horizon=24, detection_mode="direct",
    best_model="XGBoost", rmse=10.0, mae=8.0, r2=0.92,
    rows_processed=500, anomaly_count=3,
)


class TestSaveForecastRun:

    def test_returns_auto_assigned_run_id(self, db_session):
        """save_forecast_run must return a ForecastRun with an auto-assigned integer run_id."""
        run = save_forecast_run(db_session, session_id="s-1", **_RUN)
        assert isinstance(run.run_id, int)
        assert run.run_id >= 1

    def test_persists_all_fields(self, db_session):
        """Every keyword argument must appear on the returned ORM object."""
        run = save_forecast_run(
            db_session, session_id="s-2",
            filename="solar.csv", horizon=48, detection_mode="estimated",
            best_model="LightGBM", rmse=55.1, mae=38.2, r2=0.88,
            rows_processed=1000, anomaly_count=12,
        )
        assert run.session_id     == "s-2"
        assert run.filename       == "solar.csv"
        assert run.horizon        == 48
        assert run.detection_mode == "estimated"
        assert run.best_model     == "LightGBM"
        assert run.rmse           == pytest.approx(55.1)
        assert run.r2             == pytest.approx(0.88)
        assert run.anomaly_count  == 12

    def test_sets_created_at_automatically(self, db_session):
        """created_at must be set to a recent datetime without the caller providing it."""
        before = datetime.utcnow()
        run    = save_forecast_run(db_session, session_id="s-3", **_RUN)
        after  = datetime.utcnow()
        assert before <= run.created_at <= after


class TestGetRunById:

    def test_returns_correct_row(self, db_session):
        """get_run_by_id must retrieve the exact run that was saved."""
        run     = save_forecast_run(db_session, session_id="s-4", **_RUN)
        fetched = get_run_by_id(db_session, run.run_id)
        assert fetched is not None
        assert fetched.run_id   == run.run_id
        assert fetched.filename == _RUN["filename"]

    def test_returns_none_for_missing_id(self, db_session):
        """get_run_by_id must return None (not raise) when run_id does not exist."""
        assert get_run_by_id(db_session, run_id=99999) is None


class TestGetAllRuns:

    def test_returns_newest_first(self, db_session):
        """Runs must be ordered by created_at descending."""
        for sid in ("a", "b", "c"):
            save_forecast_run(db_session, session_id=sid, **_RUN)
        runs = get_all_runs(db_session)
        assert runs[0].session_id == "c"

    def test_respects_limit(self, db_session):
        """get_all_runs must honour the limit parameter."""
        for i in range(5):
            save_forecast_run(db_session, session_id=f"s-{i}", **_RUN)
        assert len(get_all_runs(db_session, limit=3)) == 3

    def test_empty_db_returns_empty_list(self, db_session):
        """get_all_runs on an empty table must return [] without raising."""
        assert get_all_runs(db_session) == []


class TestSaveForecastPoints:

    def _run(self, db_session):
        return save_forecast_run(db_session, session_id="pts", **_RUN)

    def test_inserts_all_records(self, db_session):
        """save_forecast_points must create one ForecastPoint row per dict."""
        run = self._run(db_session)
        save_forecast_points(db_session, run.run_id, [
            {"timestamp": "2024-06-01T10:00:00", "predicted": 300.0, "actual": 290.0},
            {"timestamp": "2024-06-01T11:00:00", "predicted": 400.0, "actual": 410.0},
            {"timestamp": "2024-06-01T12:00:00", "predicted": 500.0, "actual": 495.0},
        ])
        assert len(get_forecast_points_by_run_id(db_session, run.run_id)) == 3

    def test_parses_iso_timestamp_string(self, db_session):
        """ISO string timestamps must be converted to datetime objects in the DB."""
        run = self._run(db_session)
        save_forecast_points(db_session, run.run_id,
                             [{"timestamp": "2024-06-01T12:00:00", "predicted": 500.0}])
        row = get_forecast_points_by_run_id(db_session, run.run_id)[0]
        assert isinstance(row.timestamp, datetime)

    def test_accepts_datetime_object(self, db_session):
        """save_forecast_points must also accept a native datetime (not just ISO strings)."""
        run = self._run(db_session)
        save_forecast_points(db_session, run.run_id,
                             [{"timestamp": datetime(2024, 6, 1, 12), "predicted": 400.0}])
        assert len(get_forecast_points_by_run_id(db_session, run.run_id)) == 1

    def test_marks_future_flag(self, db_session):
        """is_future=True must be stored correctly on future forecast points."""
        run = self._run(db_session)
        save_forecast_points(db_session, run.run_id,
                             [{"timestamp": "2024-06-10T12:00:00", "predicted": 550.0}],
                             is_future=True)
        row = get_forecast_points_by_run_id(db_session, run.run_id)[0]
        assert row.is_future is True


class TestGetForecastPointsByRunId:

    def test_returns_empty_list_for_unknown_run(self, db_session):
        """Must return [] for a run_id with no points."""
        assert get_forecast_points_by_run_id(db_session, run_id=9999) == []

    def test_ordered_by_timestamp_ascending(self, db_session):
        """Points must be returned sorted ascending by timestamp."""
        run = save_forecast_run(db_session, session_id="ord", **_RUN)
        save_forecast_points(db_session, run.run_id, [
            {"timestamp": "2024-06-01T14:00:00", "predicted": 400.0},
            {"timestamp": "2024-06-01T10:00:00", "predicted": 200.0},
            {"timestamp": "2024-06-01T12:00:00", "predicted": 500.0},
        ])
        rows = get_forecast_points_by_run_id(db_session, run.run_id)
        ts   = [r.timestamp for r in rows]
        assert ts == sorted(ts)


class TestSaveDataset:

    def test_returns_row_with_id(self, db_session):
        """save_dataset must return a Dataset ORM object with an auto-assigned dataset_id."""
        ds = save_dataset(
            db_session,
            session_id="ds-1", filename="data.csv",
            file_path="/tmp/data.csv", file_size=1024,
            file_hash="abc123", row_count=200,
            column_map='{"irradiance": "GHI"}', detection_mode="direct",
        )
        assert isinstance(ds.dataset_id, int)
        assert ds.filename == "data.csv"


class TestForecastRunRepr:

    def test_repr_contains_key_fields(self, db_session):
        """__repr__ must include filename, horizon, and rmse."""
        run = save_forecast_run(db_session, session_id="repr", **_RUN)
        r   = repr(run)
        assert "test.csv" in r
        assert "24"       in r
        assert "10.0"     in r

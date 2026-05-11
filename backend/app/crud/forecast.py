from __future__ import annotations

from datetime import datetime

import psycopg

from app.models.forecast_point import ForecastPoint
from app.models.forecast_run import ForecastRun


def save_forecast_run(conn: psycopg.Connection, **kwargs) -> ForecastRun:
    row = conn.execute(
        """
        INSERT INTO forecast_runs
            (dataset_id, session_id, filename, horizon, detection_mode, best_model,
             rmse, mae, r2, rows_processed, anomaly_count)
        VALUES
            (%(dataset_id)s, %(session_id)s, %(filename)s, %(horizon)s,
             %(detection_mode)s, %(best_model)s,
             %(rmse)s, %(mae)s, %(r2)s, %(rows_processed)s, %(anomaly_count)s)
        RETURNING *
        """,
        {
            "dataset_id":     kwargs.get("dataset_id"),
            "session_id":     kwargs["session_id"],
            "filename":       kwargs["filename"],
            "horizon":        kwargs["horizon"],
            "detection_mode": kwargs["detection_mode"],
            "best_model":     kwargs.get("best_model"),
            "rmse":           kwargs["rmse"],
            "mae":            kwargs["mae"],
            "r2":             kwargs["r2"],
            "rows_processed": kwargs["rows_processed"],
            "anomaly_count":  kwargs.get("anomaly_count", 0),
        },
    ).fetchone()
    conn.commit()
    return ForecastRun(**row)


def get_run_by_id(conn: psycopg.Connection, run_id: int) -> ForecastRun | None:
    row = conn.execute(
        "SELECT * FROM forecast_runs WHERE run_id = %s", (run_id,)
    ).fetchone()
    return ForecastRun(**row) if row else None


def get_all_runs(conn: psycopg.Connection, limit: int = 100) -> list[ForecastRun]:
    rows = conn.execute(
        "SELECT * FROM forecast_runs ORDER BY created_at DESC LIMIT %s", (limit,)
    ).fetchall()
    return [ForecastRun(**r) for r in rows]


def save_forecast_points(
    conn: psycopg.Connection,
    run_id: int,
    points: list[dict],
    is_future: bool = False,
) -> None:
    records = []
    for p in points:
        ts = p["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        records.append((
            run_id,
            ts,
            p["predicted"],
            p.get("actual"),
            p.get("lower"),
            p.get("upper"),
            is_future,
        ))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO forecast_points
                (run_id, timestamp, predicted, actual, lower, upper, is_future)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )
    conn.commit()


def get_forecast_points_by_run_id(
    conn: psycopg.Connection, run_id: int
) -> list[ForecastPoint]:
    rows = conn.execute(
        "SELECT * FROM forecast_points WHERE run_id = %s ORDER BY timestamp ASC",
        (run_id,),
    ).fetchall()
    return [ForecastPoint(**r) for r in rows]

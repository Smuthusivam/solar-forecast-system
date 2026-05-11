"""
backend/tests/conftest.py — Shared fixtures for backend unit tests.

Requires a running PostgreSQL instance. Set TEST_DATABASE_URL in .env
(or the environment) before running tests, e.g.:
  TEST_DATABASE_URL=postgresql://user:password@localhost:5432/solar_forecast_test
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import psycopg
import pytest
from psycopg.rows import dict_row

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR    = os.path.dirname(BACKEND_DIR)
for p in (BACKEND_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

_TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "").strip()
if not _TEST_DB_URL:
    pytest.exit(
        "TEST_DATABASE_URL is not set. Add it to your .env file:\n"
        "  TEST_DATABASE_URL=postgresql://user:password@localhost:5432/solar_forecast_test",
        returncode=1,
    )


def _create_tables(conn: psycopg.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id     SERIAL PRIMARY KEY,
            session_id     VARCHAR(64)  NOT NULL,
            filename       VARCHAR(255) NOT NULL,
            file_path      VARCHAR(512) NOT NULL,
            file_size      INTEGER      NOT NULL,
            file_hash      VARCHAR(64)  NOT NULL,
            row_count      INTEGER      NOT NULL,
            column_map     TEXT         NOT NULL,
            detection_mode VARCHAR(16)  NOT NULL,
            created_at     TIMESTAMP    NOT NULL DEFAULT NOW()
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id         SERIAL PRIMARY KEY,
            dataset_id     INTEGER,
            session_id     VARCHAR(64)  NOT NULL,
            filename       VARCHAR(255) NOT NULL,
            horizon        INTEGER      NOT NULL,
            detection_mode VARCHAR(16)  NOT NULL,
            best_model     VARCHAR(16),
            rmse           DOUBLE PRECISION NOT NULL,
            mae            DOUBLE PRECISION NOT NULL,
            r2             DOUBLE PRECISION NOT NULL,
            rows_processed INTEGER      NOT NULL,
            anomaly_count  INTEGER      NOT NULL DEFAULT 0,
            created_at     TIMESTAMP    NOT NULL DEFAULT NOW()
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_points (
            point_id  SERIAL PRIMARY KEY,
            run_id    INTEGER          NOT NULL,
            timestamp TIMESTAMP        NOT NULL,
            predicted DOUBLE PRECISION NOT NULL,
            actual    DOUBLE PRECISION,
            lower     DOUBLE PRECISION,
            upper     DOUBLE PRECISION,
            is_future BOOLEAN          NOT NULL DEFAULT FALSE
        )
    """)
    conn.commit()


def _drop_tables(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS forecast_points")
    conn.execute("DROP TABLE IF EXISTS forecast_runs")
    conn.execute("DROP TABLE IF EXISTS datasets")
    conn.commit()


@pytest.fixture()
def db_session():
    """
    Fresh PostgreSQL connection for each test.
    Tables are created on setup and dropped on teardown — no state leaks.
    """
    with psycopg.connect(_TEST_DB_URL, row_factory=dict_row) as conn:
        _create_tables(conn)
        yield conn
        _drop_tables(conn)


# ── Irradiance data builders (used by test_anomaly.py) ────────────────────────

def make_irradiance_series(n: int = 96, seed: int = 42) -> pd.Series:
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2024-06-01", periods=n, freq="h")
    vals = np.array([
        max(0.0, 600.0 * (1 - abs(12 - ts.hour) / 12) + rng.normal(0, 20))
        for ts in idx
    ])
    return pd.Series(vals, index=idx, name="irradiance")


def make_irradiance_df(n: int = 96) -> pd.DataFrame:
    s = make_irradiance_series(n)
    return pd.DataFrame({"irradiance": s}, index=s.index)

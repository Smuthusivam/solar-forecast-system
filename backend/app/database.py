# Database connection and initialization logic using psycopg and connection pooling.
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not _DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Set it to a PostgreSQL connection string, e.g. "
        "postgresql://user:password@localhost:5432/solar_forecast"
    )

if _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("DATABASE_URL scheme corrected: postgres:// → postgresql://")

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        min_size = int(os.getenv("DB_POOL_MIN", "1"))
        max_size = int(os.getenv("DB_POOL_MAX", "10"))
        _pool = ConnectionPool(
            conninfo=_DATABASE_URL,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        logger.info("PostgreSQL connection pool created (min=%d, max=%d)", min_size, max_size)
    return _pool


@contextmanager
def get_db() -> Generator[psycopg.Connection, None, None]:
    pool = _get_pool()
    with pool.connection() as conn:
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise


def init_db() -> None:
    with get_db() as conn:
        conn.execute("SELECT pg_advisory_lock(1234567890)")  # one worker runs DDL at a time
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
        conn.execute("CREATE INDEX IF NOT EXISTS ix_datasets_session_id ON datasets (session_id)")

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
        conn.execute("CREATE INDEX IF NOT EXISTS ix_forecast_runs_session_id ON forecast_runs (session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_forecast_runs_dataset_id ON forecast_runs (dataset_id)")

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
        conn.execute("CREATE INDEX IF NOT EXISTS ix_forecast_points_run_id   ON forecast_points (run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_forecast_points_timestamp ON forecast_points (timestamp)")

        conn.commit()
        conn.execute("SELECT pg_advisory_unlock(1234567890)")
    logger.info("Database tables verified / created")


def get_db_dep() -> Generator[psycopg.Connection, None, None]:
    """FastAPI dependency — yields one connection per request."""
    with get_db() as conn:
        yield conn


def db_is_online() -> bool:
    try:
        with get_db() as conn:
            conn.execute("SELECT 1 FROM forecast_runs LIMIT 1")
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False

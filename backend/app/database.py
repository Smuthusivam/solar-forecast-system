"""
database.py — Database layer for the Solar Irradiance Forecasting System.

Supports two engines via a single DATABASE_URL environment variable:
  - Local development : SQLite  (automatic fallback, zero config)
  - Cloud deployment  : PostgreSQL (Railway / Render — set DATABASE_URL)

SQLAlchemy handles all dialect differences transparently.
Your ORM models, routers, and services never change between environments.

Environment variables:
  DATABASE_URL      Full connection string (set by Railway automatically
                    when you attach a Postgres plugin).
                    If not set → falls back to local SQLite.

  DB_ECHO           Set to "true" to log all SQL queries to stdout.
                    Useful during development, disable in production.

  DB_POOL_SIZE      Number of persistent connections in the pool (default 5).
                    Ignored for SQLite (no pooling needed).

  DB_MAX_OVERFLOW   Extra connections allowed above pool size (default 10).
                    Ignored for SQLite.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Connection string resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_database_url() -> tuple[str, bool]:
    """
    Returns (database_url, is_sqlite).

    Priority:
      1. DATABASE_URL env var  → PostgreSQL (or any other dialect)
      2. Fallback              → SQLite at backend/solar_forecast.db

    Railway injects DATABASE_URL as  postgres://...  (note: not postgresql://).
    SQLAlchemy requires  postgresql://  so we fix that automatically.
    """
    url = os.getenv("DATABASE_URL", "").strip()

    if url:
        # Railway uses the legacy "postgres://" scheme — SQLAlchemy rejects it
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
            logger.info("DATABASE_URL scheme corrected: postgres:// → postgresql://")

        logger.info("Using cloud database: %s", url.split("@")[-1])  # hide credentials
        return url, False

    # Local SQLite fallback
    # __file__ = backend/app/database.py  →  dirname × 2 = backend/
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path     = os.path.join(backend_dir, "solar_forecast.db")
    sqlite_url  = f"sqlite:///{db_path}"
    logger.info("DATABASE_URL not set — using SQLite at: %s", db_path)
    return sqlite_url, True


DATABASE_URL, _IS_SQLITE = _resolve_database_url()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Engine factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    """
    Builds the SQLAlchemy engine with settings appropriate for the dialect.

    SQLite:
      - check_same_thread=False   required for FastAPI's threaded request model
      - NullPool                  SQLite doesn't benefit from connection pooling
      - WAL journal mode          enables concurrent reads during a write
      - foreign_keys = ON         SQLite disables FK enforcement by default

    PostgreSQL:
      - QueuePool                 persistent connection pool (much faster than
                                  opening a new TCP connection per request)
      - pool_pre_ping             detects stale connections (cloud DBs drop idle
                                  connections after ~5 minutes of inactivity)
      - pool_size / max_overflow  tunable via env vars for horizontal scaling
    """
    echo = os.getenv("DB_ECHO", "false").lower() == "true"

    if _IS_SQLITE:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,          # no pool overhead for file-based DB
            echo=echo,
        )

        # Apply SQLite-specific pragmas on every new connection
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            # WAL = Write-Ahead Logging: readers don't block writers
            cursor.execute("PRAGMA journal_mode=WAL;")
            # Enforce foreign key constraints (off by default in SQLite)
            cursor.execute("PRAGMA foreign_keys=ON;")
            # Reduce fsync calls — safe for non-critical thesis data
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.close()

        logger.info("SQLite engine created (WAL mode, FK enforcement ON)")
        return engine

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    pool_size    = int(os.getenv("DB_POOL_SIZE",    "5"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,      # test connection health before handing it out
        pool_recycle=1800,       # recycle connections every 30 min (beats cloud timeouts)
        echo=echo,
    )
    logger.info(
        "PostgreSQL engine created (pool_size=%d, max_overflow=%d)",
        pool_size, max_overflow,
    )
    return engine


engine = _build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,   # explicit commits keep transactions predictable
    autoflush=False,    # don't flush on every query — flush only on commit
    expire_on_commit=False,  # keep ORM objects usable after commit (avoids lazy-load errors)
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ORM declarative base + models
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ForecastRun(Base):
    """
    One row per completed forecast run.

    Stores only metadata — not the forecast arrays themselves (too large).
    The History page reads from this table; the full forecast must be re-run
    to get chart data again.

    Column notes:
      session_id      UUID string from the upload session — not a FK because
                      sessions are in-memory, not persisted to DB.
      detection_mode  "direct" | "estimated"  (mirrors DetectionMode enum)
      anomaly_count   Written back after anomaly detection runs; defaults to
                      0 so the forecast endpoint can save the run before
                      anomaly detection completes.
    """
    __tablename__ = "forecast_runs"

    run_id         = Column(Integer,  primary_key=True, autoincrement=True)
    session_id     = Column(String(64),  nullable=False, index=True)
    filename       = Column(String(255), nullable=False)
    horizon        = Column(Integer,     nullable=False)           # 24 / 48 / 72
    detection_mode = Column(String(16),  nullable=False)           # "direct" / "estimated"

    # Ensemble metrics — stored so History page can show quality at a glance
    ensemble_rmse  = Column(Float, nullable=False)
    ensemble_mae   = Column(Float, nullable=False)
    ensemble_r2    = Column(Float, nullable=False)

    rows_processed = Column(Integer, nullable=False)
    anomaly_count  = Column(Integer, nullable=False, default=0)

    # ISO-8601 datetime — stored as DateTime, rendered as string in API responses
    created_at     = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<ForecastRun id={self.run_id} "
            f"file={self.filename!r} horizon={self.horizon}h "
            f"rmse={self.ensemble_rmse:.3f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FastAPI dependency — session-per-request pattern
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    Yield one SQLAlchemy Session per HTTP request, then close it.

    The  finally  block guarantees the session is closed even if the
    route handler raises an unhandled exception — no connection leaks.

    Usage in any router:

        from fastapi import Depends
        from sqlalchemy.orm import Session
        from app.database import get_db

        @router.get("/something")
        def endpoint(db: Session = Depends(get_db)):
            runs = db.query(ForecastRun).all()
            ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()   # roll back any partial writes on unhandled errors
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Initialisation — called once at startup from main.py
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables that don't already exist.

    Uses  CREATE TABLE IF NOT EXISTS  internally — safe to call on every
    startup. If you add a new column during development:
      - SQLite: delete solar_forecast.db and let it regenerate.
      - PostgreSQL: add an Alembic migration (or ALTER TABLE manually).
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Health check helper — used by GET /health in main.py
# ─────────────────────────────────────────────────────────────────────────────

def db_is_online() -> bool:
    """
    Returns True if the DB is reachable AND the forecast_runs table exists.

    Pings the actual table (not just the engine) so the health endpoint
    catches the edge case where the file exists but init_db() never ran.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM forecast_runs LIMIT 1"))
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CRUD helpers — thin wrappers used by routers
#     Keeping DB logic here (not in routers) makes unit testing trivial.
# ─────────────────────────────────────────────────────────────────────────────

def save_forecast_run(db: Session, **kwargs) -> ForecastRun:
    """
    Insert a new ForecastRun row and return the persisted object
    (with run_id populated by the DB).

    Usage:
        run = save_forecast_run(
            db,
            session_id="abc-123",
            filename="data.csv",
            horizon=24,
            detection_mode="direct",
            ensemble_rmse=45.2,
            ensemble_mae=31.1,
            ensemble_r2=0.91,
            rows_processed=8760,
        )
        print(run.run_id)   # auto-assigned integer
    """
    run = ForecastRun(**kwargs)
    db.add(run)
    db.commit()
    db.refresh(run)   # reload from DB to populate auto-generated fields
    return run


def update_anomaly_count(db: Session, run_id: int, count: int) -> None:
    """
    Write back the anomaly count after anomaly detection completes.
    Called from the anomaly router — the forecast run is saved first
    (with count=0), then updated once detection finishes.
    """
    db.query(ForecastRun).filter(ForecastRun.run_id == run_id).update(
        {"anomaly_count": count}
    )
    db.commit()


def get_all_runs(db: Session, limit: int = 100) -> list[ForecastRun]:
    """Return all forecast runs, newest first, capped at `limit`."""
    return (
        db.query(ForecastRun)
        .order_by(ForecastRun.created_at.desc())
        .limit(limit)
        .all()
    )


def get_run_by_id(db: Session, run_id: int) -> ForecastRun | None:
    """Return a single run by primary key, or None if not found."""
    return db.query(ForecastRun).filter(ForecastRun.run_id == run_id).first()
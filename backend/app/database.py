# Database layer — supports SQLite locally and PostgreSQL in production via DATABASE_URL.

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


def _resolve_database_url() -> tuple[str, bool]:
    # Use DATABASE_URL if set (Railway/Render), otherwise fall back to local SQLite.
    url = os.getenv("DATABASE_URL", "").strip()

    if url:
        # Railway uses the legacy "postgres://" scheme — SQLAlchemy needs "postgresql://"
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
            logger.info("DATABASE_URL scheme corrected: postgres:// → postgresql://")

        logger.info("Using cloud database: %s", url.split("@")[-1])
        return url, False

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path     = os.path.join(backend_dir, "storage", "db", "solar_forecast.db")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    sqlite_url = f"sqlite:///{db_path}"
    logger.info("DATABASE_URL not set — using SQLite at: %s", db_path)
    return sqlite_url, True


DATABASE_URL, _IS_SQLITE = _resolve_database_url()


def _build_engine() -> Engine:
    # Build engine with dialect-appropriate settings (SQLite vs PostgreSQL).
    echo = os.getenv("DB_ECHO", "false").lower() == "true"

    if _IS_SQLITE:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            echo=echo,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")     # readers don't block writers
            cursor.execute("PRAGMA foreign_keys=ON;")      # SQLite disables FKs by default
            cursor.execute("PRAGMA synchronous=NORMAL;")   # safe tradeoff for non-critical data
            cursor.close()

        logger.info("SQLite engine created (WAL mode, FK enforcement ON)")
        return engine

    pool_size    = int(os.getenv("DB_POOL_SIZE",    "5"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,    # test connections before handing them out
        pool_recycle=1800,     # recycle every 30 min to beat cloud idle timeouts
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
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class ForecastRun(Base):
    # One row per completed forecast run — stores metadata only, not the full arrays.
    __tablename__ = "forecast_runs"

    run_id         = Column(Integer,     primary_key=True, autoincrement=True)
    dataset_id     = Column(Integer,     nullable=True, index=True)
    session_id     = Column(String(64),  nullable=False, index=True)
    filename       = Column(String(255), nullable=False)
    horizon        = Column(Integer,     nullable=False)
    detection_mode = Column(String(16),  nullable=False)
    best_model     = Column(String(16),  nullable=True)

    rmse           = Column(Float,   nullable=False)
    mae            = Column(Float,   nullable=False)
    r2             = Column(Float,   nullable=False)

    rows_processed = Column(Integer, nullable=False)
    anomaly_count  = Column(Integer, nullable=False, default=0)

    created_at     = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<ForecastRun id={self.run_id} "
            f"file={self.filename!r} horizon={self.horizon}h "
            f"rmse={self.rmse:.3f}>"
        )


class Dataset(Base):
    # Metadata about each uploaded CSV.
    __tablename__ = "datasets"

    dataset_id     = Column(Integer,     primary_key=True, autoincrement=True)
    session_id     = Column(String(64),  nullable=False, index=True)
    filename       = Column(String(255), nullable=False)
    file_path      = Column(String(512), nullable=False)
    file_size      = Column(Integer,     nullable=False)
    file_hash      = Column(String(64),  nullable=False)
    row_count      = Column(Integer,     nullable=False)
    column_map     = Column(Text,        nullable=False)
    detection_mode = Column(String(16),  nullable=False)
    created_at     = Column(DateTime,    nullable=False, default=datetime.utcnow)


class ForecastPoint(Base):
    # Time-series output for each forecast run.
    __tablename__ = "forecast_points"

    point_id  = Column(Integer,  primary_key=True, autoincrement=True)
    run_id    = Column(Integer,  nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    predicted = Column(Float,    nullable=False)
    actual    = Column(Float,    nullable=True)
    lower     = Column(Float,    nullable=True)
    upper     = Column(Float,    nullable=True)
    is_future = Column(Boolean,  nullable=False, default=False)


def get_db() -> Generator[Session, None, None]:
    # Yield one session per request and close it afterward — no connection leaks.
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    # Create all tables that don't exist yet — safe to call on every startup.
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created")


def db_is_online() -> bool:
    # Ping the actual table, not just the engine, to catch missing init_db() calls.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM forecast_runs LIMIT 1"))
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False


def save_forecast_run(db: Session, **kwargs) -> ForecastRun:
    # Insert a new ForecastRun and return it with the auto-assigned run_id.
    run = ForecastRun(**kwargs)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def save_dataset(db: Session, **kwargs) -> Dataset:
    dataset = Dataset(**kwargs)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def save_forecast_points(db: Session, run_id: int, points: list[dict], is_future: bool = False) -> None:
    records = []
    for p in points:
        records.append(
            ForecastPoint(
                run_id=run_id,
                timestamp=p["timestamp"],
                predicted=p["predicted"],
                actual=p.get("actual"),
                lower=p.get("lower"),
                upper=p.get("upper"),
                is_future=is_future,
            )
        )
    db.add_all(records)
    db.commit()


def update_anomaly_count(db: Session, run_id: int, count: int) -> None:
    # Write back the anomaly count once detection finishes (run is saved first with 0).
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

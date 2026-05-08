"""
backend/tests/conftest.py — Shared fixtures for backend unit tests.

Provides the db_session fixture and irradiance data builders used by
test_database.py and test_anomaly.py.
ML preprocessing fixtures live in ml_core/tests/conftest.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR    = os.path.dirname(BACKEND_DIR)
for p in (BACKEND_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.database import Base


@pytest.fixture()
def db_session():
    """
    In-memory SQLite session, created fresh for each test.
    Tables are created on setup and dropped on teardown — no state leaks between tests.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


# ── Irradiance data builders (used by test_anomaly.py) ────────────────────────

def make_irradiance_series(n: int = 96, seed: int = 42) -> pd.Series:
    """
    Return a pd.Series with a DatetimeIndex and realistic daytime irradiance values.
    """
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2024-06-01", periods=n, freq="h")
    vals = np.array([
        max(0.0, 600.0 * (1 - abs(12 - ts.hour) / 12) + rng.normal(0, 20))
        for ts in idx
    ])
    return pd.Series(vals, index=idx, name="irradiance")


def make_irradiance_df(n: int = 96) -> pd.DataFrame:
    """Return a DataFrame suitable for detect_anomalies — 'irradiance' column + DatetimeIndex."""
    s = make_irradiance_series(n)
    return pd.DataFrame({"irradiance": s}, index=s.index)

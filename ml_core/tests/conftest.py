"""
ml_core/tests/conftest.py — Shared fixtures and data builders for ml_core tests.

Provides CSV byte builders and irradiance Series/DataFrame factories.
No database, no FastAPI — pure ML/data concerns only.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ML_CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ML_CORE_ROOT)
for p in (ML_CORE_ROOT, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── CSV builders ──────────────────────────────────────────────────────────────

def make_csv_bytes(rows: int = 72, with_irradiance: bool = True) -> bytes:
    """
    Build a minimal well-formed CSV in memory.

    Parameters
    ----------
    rows            Number of hourly data rows (≥48 satisfies MIN_ROWS).
    with_irradiance Include an irradiance column; set False to test GHI estimation path.
    """
    buf  = io.StringIO()
    w    = csv.writer(buf)
    cols = ["timestamp", "temperature", "humidity", "wind_speed", "cloud_cover"]
    if with_irradiance:
        cols.insert(1, "irradiance")
    w.writerow(cols)

    start = datetime(2024, 6, 1, 0, 0)
    for i in range(rows):
        ts   = start + timedelta(hours=i)
        hour = ts.hour
        irr  = max(0.0, 600.0 * (1 - abs(12 - hour) / 12))
        row  = [ts.isoformat(), round(20 + hour * 0.1, 2), 55.0, 3.0, 20.0]
        if with_irradiance:
            row.insert(1, round(irr, 2))
        w.writerow(row)

    return buf.getvalue().encode()


# ── Series / DataFrame builders ───────────────────────────────────────────────

def make_irradiance_series(n: int = 96, seed: int = 42) -> pd.Series:
    """
    Return a pd.Series with a DatetimeIndex and realistic daytime irradiance values.
    Used as input to the statistical anomaly detectors.
    """
    rng   = np.random.default_rng(seed)
    idx   = pd.date_range("2024-06-01", periods=n, freq="h")
    vals  = np.array([
        max(0.0, 600.0 * (1 - abs(12 - ts.hour) / 12) + rng.normal(0, 20))
        for ts in idx
    ])
    return pd.Series(vals, index=idx, name="irradiance")


def make_irradiance_df(n: int = 96) -> pd.DataFrame:
    """Return a DataFrame suitable for detect_anomalies — 'irradiance' column + DatetimeIndex."""
    s = make_irradiance_series(n)
    return pd.DataFrame({"irradiance": s}, index=s.index)

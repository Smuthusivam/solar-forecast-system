from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# Represents a single forecast point with predicted and actual values
@dataclass
class ForecastPoint:
    point_id:  int
    run_id:    int
    timestamp: datetime
    predicted: float
    is_future: bool
    actual:    float | None = None
    lower:     float | None = None
    upper:     float | None = None

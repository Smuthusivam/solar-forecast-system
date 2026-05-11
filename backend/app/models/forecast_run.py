from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ForecastRun:
    run_id:         int
    session_id:     str
    filename:       str
    horizon:        int
    detection_mode: str
    rmse:           float
    mae:            float
    r2:             float
    rows_processed: int
    anomaly_count:  int
    created_at:     datetime
    dataset_id:     int | None = None
    best_model:     str | None = None

    def __repr__(self) -> str:
        return (
            f"<ForecastRun id={self.run_id} "
            f"file={self.filename!r} horizon={self.horizon}h "
            f"rmse={self.rmse:.3f}>"
        )

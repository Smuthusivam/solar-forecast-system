# GET /api/export/csv — download forecast results as CSV.

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session
from ml_core.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

_BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_EXPORTS_DIR = os.path.join(_BACKEND_DIR, "storage", "exports")
os.makedirs(_EXPORTS_DIR, exist_ok=True)


@router.get("/export/csv")
def export_csv(session_id: str, horizon: int = 24):
    session = get_session(session_id)
    df      = session["df"]
    col_map = session["detected_cols"]

    try:
        result = run_pipeline(df, col_map, horizon=horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    rows = [
        {
            "timestamp": p["timestamp"],
            "predicted": p["predicted"],
            "actual":    p["actual"],
            "lower":     p["lower"],
            "upper":     p["upper"],
        }
        for p in result["forecast"]
    ]
    export_df = pd.DataFrame(rows)

    buffer    = io.StringIO()
    export_df.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode()

    filename = f"solar_forecast_{session_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(_EXPORTS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(csv_bytes)
    logger.info("Export saved: %s", filepath)

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

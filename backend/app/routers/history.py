# FastAPI router for retrieving forecast history.

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db_dep
from app.crud import get_all_runs, get_run_by_id, get_forecast_points_by_run_id
from app.models.schemas import ForecastRunSummary, HistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Return all past forecast runs, newest first.
@router.get("/history", response_model=HistoryResponse)
def get_history(limit: int = 100, db=Depends(get_db_dep)):
    runs = get_all_runs(db, limit=limit)

    return HistoryResponse(
        total_runs = len(runs),
        runs       = [ForecastRunSummary.model_validate(r) for r in runs],
    )

# Return a single forecast run by ID.
@router.get("/history/{run_id}", response_model=ForecastRunSummary)
def get_run(run_id: int, db=Depends(get_db_dep)):
    run = get_run_by_id(db, run_id)

    if not run:
        raise HTTPException(
            status_code=404,
            detail={
                "code":    "run_not_found",
                "message": f"Forecast run {run_id} not found.",
                "field":   "run_id",
            },
        )

    return ForecastRunSummary.model_validate(run)

# Return all stored forecast points for a run (predicted vs actual + confidence bands).
@router.get("/history/{run_id}/points")
def get_run_points(run_id: int, db=Depends(get_db_dep)):
    run = get_run_by_id(db, run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "run_not_found", "message": f"Forecast run {run_id} not found.", "field": "run_id"},
        )

    points = get_forecast_points_by_run_id(db, run_id)
    return {
        "run_id":   run_id,
        "count":    len(points),
        "points": [
            {
                "timestamp": p.timestamp if isinstance(p.timestamp, str) else p.timestamp.isoformat(),
                "predicted": p.predicted,
                "lower":     p.lower,
                "upper":     p.upper,
                "is_future": p.is_future,
            }
            for p in points
        ],
    }

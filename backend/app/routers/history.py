# GET /api/history and GET /api/history/{run_id}.

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, get_all_runs, get_run_by_id, get_forecast_points_by_run_id
from app.models.schemas import ForecastRunSummary, HistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    """Return all past forecast runs, newest first."""
    runs = get_all_runs(db, limit=limit)

    return HistoryResponse(
        total_runs = len(runs),
        runs       = [ForecastRunSummary.model_validate(r) for r in runs],
    )


@router.get("/history/{run_id}", response_model=ForecastRunSummary)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Return a single forecast run by ID."""
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


@router.get("/history/{run_id}/points")
def get_run_points(run_id: int, db: Session = Depends(get_db)):
    """Return all stored forecast points for a run (predicted vs actual + confidence bands)."""
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
                "actual":    p.actual,
                "lower":     p.lower,
                "upper":     p.upper,
                "is_future": p.is_future,
            }
            for p in points
        ],
    }

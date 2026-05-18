# POST /api/forecast/run.

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db_dep
from app.crud import save_forecast_run, save_forecast_points
from app.models.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelMetrics,
    DetectionMode,
    PerModelInfo,
    ForecastPoint,
)
from app.routers.upload import get_session
from ml_core.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/forecast/run", response_model=ForecastResponse, summary="Run Forecast on Original Data")
def run_forecast(request: ForecastRequest, db=Depends(get_db_dep)):
    session        = get_session(request.session_id)
    df             = session["df"]
    col_map        = session["detected_cols"]
    detection_mode = session["detection_mode"]
    filename       = session["filename"]

    logger.info(
        "Forecast run started | session=%s | horizon=%dh | rows=%d",
        request.session_id, request.horizon, len(df),
    )

    try:
        result = run_pipeline(
            df             = df,
            col_map        = col_map,
            horizon        = int(request.horizon),
            detection_mode = detection_mode,
            train_size     = request.train_size,
            skip_future    = request.skip_future,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code":    "pipeline_failed",
                "message": f"ML pipeline failed: {exc}",
                "field":   None,
            },
        )

    m = result["metrics"]
    try:
        db_run = save_forecast_run(
            db,
            session_id     = request.session_id,
            filename       = filename,
            horizon        = int(request.horizon),
            detection_mode = detection_mode,
            best_model     = result["best_model"],
            rmse           = m["rmse"],
            mae            = m["mae"],
            r2             = m["r2"],
            rows_processed = result["rows_processed"],
            anomaly_count  = 0,
        )
        run_id = db_run.run_id
        logger.info("Forecast run saved to DB: run_id=%d", run_id)
        if result.get("future_forecast"):
            save_forecast_points(db, run_id, result["future_forecast"], is_future=True)
    except Exception as exc:
        logger.error("DB save failed (non-fatal): %s", exc)
        run_id = -1

    forecast_points = [ForecastPoint(**p) for p in result["forecast"]]
    future_points   = [ForecastPoint(**p) for p in result.get("future_forecast", [])]

    models_info = [
        PerModelInfo(
            model_name    = mi["model_name"],
            predictions   = mi["predictions"],
            metrics       = ModelMetrics(**mi["metrics"]),
            train_metrics = ModelMetrics(**mi["train_metrics"]) if mi.get("train_metrics") else None,
            is_best       = mi["is_best"],
        )
        for mi in result["models_info"]
    ]

    return ForecastResponse(
        session_id         = request.session_id,
        run_id             = run_id,
        horizon            = int(request.horizon),
        detection_mode     = DetectionMode(detection_mode),
        best_model         = result["best_model"],
        forecast           = forecast_points,
        future_forecast    = future_points,
        metrics            = ModelMetrics(**m),
        rows_processed     = result["rows_processed"],
        models_info        = models_info,
        feature_importance = result.get("feature_importance"),
    )

# POST /api/forecast/run.

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

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
def run_forecast(request: ForecastRequest):
    session        = get_session(request.session_id)
    df             = session["df"]
    col_map        = session["detected_cols"]
    detection_mode = session["detection_mode"]

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
        run_id             = -1,
        horizon            = int(request.horizon),
        detection_mode     = DetectionMode(detection_mode),
        best_model         = result["best_model"],
        forecast           = forecast_points,
        future_forecast    = future_points,
        metrics            = ModelMetrics(**result["metrics"]),
        rows_processed     = result["rows_processed"],
        models_info        = models_info,
        feature_importance = result.get("feature_importance"),
    )

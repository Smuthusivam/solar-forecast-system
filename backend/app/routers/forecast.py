"""
forecast.py — POST /api/forecast/run
             GET  /api/forecast/models
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, save_forecast_run
from app.models.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelMetrics,
    ForecastHorizon,
    DetectionMode,
    PerModelForecast,
    ForecastPoint,
)
from app.routers.upload import get_session
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/forecast/run", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    """
    Run the full ML pipeline on a previously uploaded dataset.

    Steps:
      1. Load session (DataFrame + metadata)
      2. Run XGBoost + LightGBM + Prophet + ensemble
      3. Save run metadata to SQLite
      4. Return full forecast response
    """
    # ── Load session ──────────────────────────────────────────────────────────
    session = get_session(request.session_id)
    df             = session["df"]
    col_map        = session["detected_cols"]
    detection_mode = session["detection_mode"]
    filename       = session["filename"]

    logger.info(
        "Forecast run started | session=%s | horizon=%dh | rows=%d",
        request.session_id, request.horizon, len(df),
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_pipeline(
            df             = df,
            col_map        = col_map,
            horizon        = int(request.horizon),
            detection_mode = detection_mode,
            train_size     = request.train_size,
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

    # ── Save to database ──────────────────────────────────────────────────────
    em = result["ensemble_metrics"]
    try:
        db_run = save_forecast_run(
            db,
            session_id     = request.session_id,
            filename       = filename,
            horizon        = int(request.horizon),
            detection_mode = detection_mode,
            ensemble_rmse  = em["rmse"],
            ensemble_mae   = em["mae"],
            ensemble_r2    = em["r2"],
            rows_processed = result["rows_processed"],
            anomaly_count  = 0,   # updated later by anomaly router
        )
        run_id = db_run.run_id
        logger.info("Forecast run saved to DB: run_id=%d", run_id)
    except Exception as exc:
        logger.error("DB save failed (non-fatal): %s", exc)
        run_id = -1   # don't crash the whole response over a DB write failure

    # ── Build response ────────────────────────────────────────────────────────
    forecast_points = [ForecastPoint(**p) for p in result["forecast"]]

    per_model = [
        PerModelForecast(
            model_name    = m["model_name"],
            predictions   = m["predictions"],
            metrics       = ModelMetrics(**m["metrics"]),
            train_metrics = ModelMetrics(**m["train_metrics"]) if m.get("train_metrics") else None,
            weight        = m["weight"],
        )
        for m in result["per_model"]
    ]

    return ForecastResponse(
        session_id        = request.session_id,
        run_id            = run_id,
        horizon           = int(request.horizon),
        detection_mode    = DetectionMode(detection_mode),
        forecast          = forecast_points,
        ensemble_metrics  = ModelMetrics(**em),
        per_model         = per_model,
        feature_importance= result.get("feature_importance"),
    )


@router.get("/forecast/models")
def list_models():
    """Return available models and their descriptions."""
    return {
        "models": [
            {
                "name":        "XGBoost",
                "type":        "Gradient Boosted Trees",
                "description": "Fast, accurate tree ensemble. Uses full feature matrix.",
            },
            {
                "name":        "LightGBM",
                "type":        "Leaf-wise Gradient Boosting",
                "description": "Microsoft's faster XGBoost alternative. Same feature matrix.",
            },
            {
                "name":        "Prophet",
                "type":        "Additive Seasonal Decomposition",
                "description": "Meta's time series model. Handles daily + annual seasonality.",
            },
            {
                "name":        "Ensemble",
                "type":        "Weighted Average",
                "description": "RMSE-weighted combination of all three models.",
            },
        ]
    }
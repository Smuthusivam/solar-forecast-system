# POST /api/forecast/run and async polling endpoints.

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db, save_forecast_points, save_forecast_run
from app.models.schemas import (
    DetectionMode,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelMetrics,
    PerModelInfo,
)
from app.routers.upload import get_session
from ml_core.anomaly import detect_anomalies
from ml_core.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


_job_lock = threading.Lock()
_forecast_jobs: dict[str, dict[str, Any]] = {}


def _serialize_response_payload(request: ForecastRequest, db: Session) -> dict[str, Any]:
    # Shared execution path used by sync endpoint and background worker.
    session = get_session(request.session_id)
    df = session["df"]
    col_map = session["detected_cols"]
    detection_mode = session["detection_mode"]
    filename = session["filename"]

    logger.info(
        "Forecast run started | session=%s | horizon=%dh | rows=%d",
        request.session_id,
        request.horizon,
        len(df),
    )

    result = run_pipeline(
        df=df,
        col_map=col_map,
        horizon=int(request.horizon),
        detection_mode=detection_mode,
        train_size=request.train_size,
        skip_future=request.skip_future,
    )

    try:
        anomaly_count = detect_anomalies(df)["anomaly_count"]
    except Exception:
        anomaly_count = 0

    m = result["metrics"]
    try:
        db_run = save_forecast_run(
            db,
            session_id=request.session_id,
            filename=filename,
            horizon=int(request.horizon),
            detection_mode=detection_mode,
            best_model=result["best_model"],
            rmse=m["rmse"],
            mae=m["mae"],
            r2=m["r2"],
            rows_processed=result["rows_processed"],
            anomaly_count=anomaly_count,
        )
        run_id = db_run.run_id
        try:
            save_forecast_points(db, run_id, result["forecast"], is_future=False)
            if result.get("future_forecast"):
                save_forecast_points(db, run_id, result["future_forecast"], is_future=True)
        except Exception as exc:
            logger.error("Forecast points save failed (non-fatal): %s", exc)
    except Exception as exc:
        logger.error("DB save failed (non-fatal): %s", exc)
        run_id = -1

    forecast_points = [ForecastPoint(**p).model_dump(mode="json") for p in result["forecast"]]
    future_points = [ForecastPoint(**p).model_dump(mode="json") for p in result.get("future_forecast", [])]

    models_info = [
        PerModelInfo(
            model_name=mi["model_name"],
            predictions=mi["predictions"],
            metrics=ModelMetrics(**mi["metrics"]),
            train_metrics=ModelMetrics(**mi["train_metrics"]) if mi.get("train_metrics") else None,
            is_best=mi["is_best"],
        ).model_dump(mode="json")
        for mi in result["models_info"]
    ]

    response_payload = {
        "session_id": request.session_id,
        "run_id": run_id,
        "horizon": int(request.horizon),
        "detection_mode": DetectionMode(detection_mode).value,
        "best_model": result["best_model"],
        "forecast": forecast_points,
        "future_forecast": future_points,
        "metrics": ModelMetrics(**m).model_dump(mode="json"),
        "models_info": models_info,
        "feature_importance": result.get("feature_importance"),
    }
    return response_payload


def _run_forecast_job(request_data: dict[str, Any]) -> None:
    # Background worker for async forecast runs.
    request = ForecastRequest(**request_data)
    with _job_lock:
        _forecast_jobs[request.session_id] = {
            "session_id": request.session_id,
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat(),
            "error": None,
            "result": None,
        }

    db = SessionLocal()
    try:
        payload = _serialize_response_payload(request, db)
        with _job_lock:
            _forecast_jobs[request.session_id] = {
                "session_id": request.session_id,
                "status": "completed",
                "updated_at": datetime.utcnow().isoformat(),
                "error": None,
                "result": payload,
            }
    except Exception as exc:
        logger.error("Background forecast failed for %s: %s", request.session_id, exc)
        with _job_lock:
            _forecast_jobs[request.session_id] = {
                "session_id": request.session_id,
                "status": "failed",
                "updated_at": datetime.utcnow().isoformat(),
                "error": str(exc),
                "result": None,
            }
    finally:
        db.close()


@router.post("/forecast/start")
def start_forecast_job(request: ForecastRequest, background_tasks: BackgroundTasks):
    # Start forecast processing in the background and return immediately.
    get_session(request.session_id)

    with _job_lock:
        current = _forecast_jobs.get(request.session_id)
        if current and current.get("status") == "processing":
            return {
                "session_id": request.session_id,
                "status": "processing",
                "message": "A forecast job is already running for this session.",
            }

        _forecast_jobs[request.session_id] = {
            "session_id": request.session_id,
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat(),
            "error": None,
            "result": None,
        }

    background_tasks.add_task(_run_forecast_job, request.model_dump())
    return {
        "session_id": request.session_id,
        "status": "processing",
        "message": "Forecast job accepted. Poll /api/status/{session_id} for updates.",
    }


@router.get("/status/{session_id}")
def get_forecast_status(session_id: str):
    # Poll this endpoint to track async forecast jobs.
    with _job_lock:
        job = _forecast_jobs.get(session_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "job_not_found",
                "message": "No forecast job found for this session.",
                "field": "session_id",
            },
        )

    return job


@router.post("/forecast/run", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    # Synchronous forecast endpoint kept for backward compatibility.
    try:
        payload = _serialize_response_payload(request, db)
        return ForecastResponse(**payload)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pipeline_failed",
                "message": f"ML pipeline failed: {exc}",
                "field": None,
            },
        )



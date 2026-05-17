"""
correction.py — FastAPI router
Endpoints:
  POST /api/correction/run/{dataset_id}   → detect + AI-correct + compare models
  GET  /api/correction/log/{session_id}   → get correction log JSON
"""

import asyncio
import os
import pickle
import uuid
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db_dep
from app.crud import save_forecast_run, save_forecast_points
from app.routers.upload import get_session, _SESSIONS_DIR
from ml_core.anomaly import detect_anomalies
from app.services.anomaly_corrector import _correct_all_async, compute_correction_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correction", tags=["Correction"])

_CORRECTION_DIR = os.path.join(os.path.dirname(_SESSIONS_DIR), "corrections")
os.makedirs(_CORRECTION_DIR, exist_ok=True)


def _correction_path(session_id: str) -> str:
    return os.path.join(_CORRECTION_DIR, f"{session_id}.pkl")


def _write_correction(session_id: str, data: dict) -> None:
    with open(_correction_path(session_id), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_correction(session_id: str) -> dict | None:
    path = _correction_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load correction session %s: %s", session_id, exc)
        return None


def register_upload_session(dataset_id: str, df: pd.DataFrame, column_map: dict):
    """Called by the upload router after preprocessing — no-op now since upload sessions are disk-backed."""
    pass


@router.post("/run/{dataset_id}")
async def run_correction(dataset_id: str):
    # Load upload session from disk (shared across workers)
    try:
        upload_session = get_session(dataset_id)
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload a CSV first via /api/upload."
        )

    df_original: pd.DataFrame = upload_session["df"]
    column_map: dict = upload_session["detected_cols"]

    # Step 1 — Detect anomalies
    detection_result = detect_anomalies(df_original)
    anomalies = detection_result["anomalies"]

    if not anomalies:
        return {
            "dataset_id": dataset_id,
            "message": "No anomalies detected — your dataset looks clean.",
            "anomaly_count": 0,
            "correction_log": [],
            "stats": {},
        }

    # Step 2 — AI correction
    df_corrected, correction_log = await _correct_all_async(
        df_original, anomalies, column_map
    )
    stats = compute_correction_stats(correction_log)

    # Step 3 — Persist correction session to disk so any worker can read it
    session_id = str(uuid.uuid4())
    _write_correction(session_id, {
        "df_corrected":   df_corrected,
        "correction_log": correction_log,
        "dataset_id":     dataset_id,
        "column_map":     column_map,
    })

    return {
        "dataset_id":         dataset_id,
        "session_id":         session_id,
        "anomaly_count":      len(anomalies),
        "anomalies_corrected": len(correction_log),
        "stats":              stats,
        "correction_log":     correction_log,
    }


@router.post("/forecast/{correction_session_id}", summary="Run Forecast on AI-Corrected Data")
async def run_forecast_from_corrected(
    correction_session_id: str,
    horizon: int = 24,
    train_size: int = 80,
    db=Depends(get_db_dep),
):
    """Run the ML forecast pipeline on the AI-corrected dataframe and save to DB."""
    session = _read_correction(correction_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Correction session not found or has expired. Please rerun correction.")

    from ml_core.pipeline import run_pipeline

    df_corrected = session["df_corrected"]
    dataset_id   = session["dataset_id"]

    try:
        upload_session  = get_session(dataset_id)
        col_map         = upload_session["detected_cols"]
        detection_mode  = upload_session["detection_mode"]
        filename        = upload_session["filename"]
    except HTTPException:
        col_map        = session.get("column_map", {})
        detection_mode = "direct"
        filename       = "corrected_dataset.csv"

    try:
        result = await asyncio.to_thread(
            run_pipeline,
            df_corrected,
            col_map,
            horizon,
            detection_mode,
            train_size,
            False,
        )
    except Exception as exc:
        logger.error("Corrected forecast pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Forecast pipeline failed: {exc}")

    # Save corrected forecast run to DB
    m = result["metrics"]
    try:
        db_run = save_forecast_run(
            db,
            session_id     = dataset_id,
            filename       = filename,
            horizon        = horizon,
            detection_mode = detection_mode,
            best_model     = result["best_model"],
            rmse           = m["rmse"],
            mae            = m["mae"],
            r2             = m["r2"],
            rows_processed = result["rows_processed"],
            anomaly_count  = 0,
        )
        run_id = db_run.run_id
        logger.info("Corrected forecast run saved to DB: run_id=%d", run_id)
        try:
            save_forecast_points(db, run_id, result["forecast"], is_future=False)
            if result.get("future_forecast"):
                save_forecast_points(db, run_id, result["future_forecast"], is_future=True)
        except Exception as exc:
            logger.error("Forecast points save failed (non-fatal): %s", exc)
    except Exception as exc:
        logger.error("DB save failed (non-fatal): %s", exc)
        run_id = -1

    forecast_points = [
        {"timestamp": p["timestamp"], "predicted": p["predicted"],
         "actual": p.get("actual"), "lower": p.get("lower"), "upper": p.get("upper")}
        for p in result["forecast"]
    ]
    future_points = [
        {"timestamp": p["timestamp"], "predicted": p["predicted"],
         "actual": p.get("actual"), "lower": p.get("lower"), "upper": p.get("upper")}
        for p in result.get("future_forecast", [])
    ]

    return {
        "session_id":            dataset_id,
        "correction_session_id": correction_session_id,
        "run_id":                run_id,
        "horizon":               horizon,
        "detection_mode":        detection_mode,
        "best_model":            result["best_model"],
        "forecast":              forecast_points,
        "future_forecast":       future_points,
        "metrics":               result["metrics"],
        "models_info":           result["models_info"],
        "feature_importance":    result.get("feature_importance"),
        "rows_processed":        result.get("rows_processed", 0),
        "source":                "corrected",
    }


@router.get("/log/{session_id}")
async def get_correction_log(session_id: str):
    session = _read_correction(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "dataset_id": session["dataset_id"],
        "corrections": session["correction_log"],
        "total": len(session["correction_log"]),
    }



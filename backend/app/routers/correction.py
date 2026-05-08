"""
correction.py — FastAPI router
Endpoints:
  POST /api/correction/run/{dataset_id}   → detect + AI-correct + compare models
  GET  /api/correction/export/{session_id} → download corrected CSV
  GET  /api/correction/log/{session_id}   → get correction log JSON
"""

import asyncio
import io
import uuid
import logging
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session
from ml_core.anomaly import detect_anomalies
from app.services.anomaly_corrector import _correct_all_async, compute_correction_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correction", tags=["correction"])

# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store  (replace with Redis / DB in production)
# ─────────────────────────────────────────────────────────────────────────────
upload_sessions: Dict[str, Any] = {}   # populated by upload router
correction_sessions: Dict[str, Any] = {}


def register_upload_session(dataset_id: str, df: pd.DataFrame, column_map: dict):
    """Called by the upload router after preprocessing."""
    upload_sessions[dataset_id] = {"df": df, "column_map": column_map}


@router.post("/run/{dataset_id}")
async def run_correction(dataset_id: str):
    """
    Full correction pipeline:
    1. Load dataset from upload session
    2. Detect anomalies
    3. AI-correct each anomaly
    4. Run ML models on original + corrected data
    5. Return side-by-side comparison
    """
    session = upload_sessions.get(dataset_id)
    if not session:
        try:
            upload_session = get_session(dataset_id)
            session = {
                "df": upload_session["df"],
                "column_map": upload_session["detected_cols"],
            }
            upload_sessions[dataset_id] = session
        except HTTPException:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found. Please upload a CSV first via /api/upload."
            )

    df_original: pd.DataFrame = session["df"]
    column_map: dict = session["column_map"]

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

    # Step 2 — AI correction (concurrent async calls)
    df_corrected, correction_log = await _correct_all_async(
        df_original, anomalies, column_map
    )
    stats = compute_correction_stats(correction_log)

    # Step 3 — Store session for export
    session_id = str(uuid.uuid4())
    correction_sessions[session_id] = {
        "df_corrected": df_corrected,
        "correction_log": correction_log,
        "dataset_id": dataset_id,
    }

    return {
        "dataset_id": dataset_id,
        "session_id": session_id,
        "anomaly_count": len(anomalies),
        "anomalies_corrected": len(correction_log),
        "stats": stats,
        "correction_log": correction_log,
    }


@router.post("/forecast/{correction_session_id}")
async def run_forecast_from_corrected(
    correction_session_id: str,
    horizon: int = 24,
    train_size: int = 80,
):
    """Run the ML forecast pipeline on the AI-corrected dataframe."""
    session = correction_sessions.get(correction_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Correction session not found or has expired. Please rerun correction.")

    from ml_core.pipeline import run_pipeline
    from app.models.schemas import ForecastPoint, PerModelInfo, ModelMetrics, DetectionMode
    import asyncio

    df_corrected = session["df_corrected"]
    dataset_id = session["dataset_id"]

    try:
        upload_session = get_session(dataset_id)
        col_map = upload_session["detected_cols"]
        detection_mode = upload_session["detection_mode"]
        filename = upload_session["filename"]
    except HTTPException:
        col_map = session.get("column_map", {})
        detection_mode = "direct"
        filename = dataset_id

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
        "session_id": dataset_id,
        "correction_session_id": correction_session_id,
        "run_id": -1,
        "horizon": horizon,
        "detection_mode": detection_mode,
        "best_model": result["best_model"],
        "forecast": forecast_points,
        "future_forecast": future_points,
        "metrics": result["metrics"],
        "models_info": result["models_info"],
        "feature_importance": result.get("feature_importance"),
        "rows_processed": result.get("rows_processed", 0),
        "source": "corrected",
    }


@router.get("/log/{session_id}")
async def get_correction_log(session_id: str):
    """Return the full correction log for a completed correction session."""
    session = correction_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "dataset_id": session["dataset_id"],
        "corrections": session["correction_log"],
        "total": len(session["correction_log"]),
    }


@router.get("/export/{session_id}")
async def export_corrected_csv(session_id: str):
    """Download the AI-corrected dataset as a CSV file."""
    session = correction_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    df: pd.DataFrame = session["df_corrected"]
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    filename = f"corrected_{session['dataset_id']}.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
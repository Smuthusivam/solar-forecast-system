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
from app.services.anomaly import detect_anomalies
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


def _metrics_delta(orig: dict, corr: dict) -> dict:
    """Compute absolute and percentage deltas between two metrics dicts."""
    delta = {}
    for key in orig:
        if key in corr and isinstance(orig[key], (int, float)):
            o, c = float(orig[key]), float(corr[key])
            delta[key] = {
                "original": round(o, 6),
                "corrected": round(c, 6),
                "delta": round(c - o, 6),
                "improvement_pct": round((o - c) / o * 100, 2) if o != 0 else 0,
            }
    return delta


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
                detail=f"Dataset '{dataset_id}' not found. Upload a CSV first via /api/upload."
            )

    df_original: pd.DataFrame = session["df"]
    column_map: dict = session["column_map"]

    # Step 1 — Detect anomalies
    detection_result = detect_anomalies(df_original)
    anomalies = detection_result["anomalies"]

    if not anomalies:
        return {
            "dataset_id": dataset_id,
            "message": "No anomalies detected — dataset is clean.",
            "anomaly_count": 0,
            "correction_log": [],
            "stats": {},
        }

    # Step 2 — AI correction (concurrent async calls)
    df_corrected, correction_log = await _correct_all_async(
        df_original, anomalies, column_map
    )
    stats = compute_correction_stats(correction_log)

    # Step 3 — Run ML pipeline on both versions in parallel
    try:
        from app.services.pipeline import run_pipeline

        logger.info("Running original + corrected pipelines in parallel...")
        results_original, results_corrected = await asyncio.gather(
            asyncio.to_thread(run_pipeline, df_original,  column_map),
            asyncio.to_thread(run_pipeline, df_corrected, column_map),
        )

        metrics_comparison = _metrics_delta(
            results_original["metrics"],
            results_corrected["metrics"],
        )

        forecasts = {
            "timestamps": [p["timestamp"] for p in results_original["forecast"]],
            "original":   [p["predicted"]  for p in results_original["forecast"]],
            "corrected":  [p["predicted"]  for p in results_corrected["forecast"]],
            "actuals":    [p["actual"]      for p in results_original["forecast"]],
        }

        model_comparison = {
            "original":  {m["model_name"]: m["metrics"] for m in results_original["models_info"]},
            "corrected": {m["model_name"]: m["metrics"] for m in results_corrected["models_info"]},
        }

        logger.info(
            "Pipeline comparison done | original RMSE=%.2f corrected RMSE=%.2f",
            results_original["metrics"]["rmse"],
            results_corrected["metrics"]["rmse"],
        )

    except Exception as e:
        logger.error(f"Pipeline comparison failed: {e}", exc_info=True)
        metrics_comparison = {}
        forecasts = {}
        model_comparison = {}

    # Step 4 — Store session for export
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
        "metrics_comparison": metrics_comparison,
        "forecasts": forecasts,
        "model_comparison": model_comparison,
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
"""
upload.py — POST /api/upload

Receives a CSV file, runs AI column detection + preprocessing,
stores the result in memory (session store), returns metadata
to the frontend for preview before the user triggers a forecast run.

Session store:
  A simple in-memory dict keyed by session_id (UUID).
  Stores the clean DataFrame + detection metadata so the forecast
  router doesn't need to re-upload and re-preprocess the file.
  
  Sessions expire after 2 hours to prevent memory leaks.
  For a thesis demo this is perfectly sufficient — no Redis needed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DetectedColumns, DetectionMode, UploadResponse
from app.services.ai_detector import detect_columns
from app.services.preprocessing import preprocess

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────────────────────────────────────

SESSION_TTL_HOURS = 2

# Structure:
# {
#   session_id: {
#     "df":             pd.DataFrame,
#     "detected_cols":  dict,
#     "detection_mode": str,
#     "filename":       str,
#     "meta":           dict,
#     "expires_at":     datetime,
#   }
# }
_sessions: dict[str, dict[str, Any]] = {}


def get_session(session_id: str) -> dict[str, Any]:
    """
    Retrieve a session by ID.
    Raises 404 if not found, 410 if expired.
    Called by forecast, anomaly, and export routers.
    """
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail={
                "code":    "session_not_found",
                "message": f"Session '{session_id}' not found. Please upload a file first.",
                "field":   "session_id",
            },
        )

    if datetime.utcnow() > session["expires_at"]:
        _sessions.pop(session_id, None)
        raise HTTPException(
            status_code=410,
            detail={
                "code":    "session_expired",
                "message": "Your session has expired. Please upload the file again.",
                "field":   "session_id",
            },
        )

    return session


def _purge_expired_sessions() -> None:
    """Remove expired sessions to prevent memory growth."""
    now     = datetime.utcnow()
    expired = [sid for sid, s in _sessions.items() if now > s["expires_at"]]
    for sid in expired:
        _sessions.pop(sid, None)
    if expired:
        logger.info("Purged %d expired sessions", len(expired))


# ─────────────────────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and detect its column structure.

    Steps:
      1. Read file bytes
      2. Quick parse to get column names + sample rows
      3. Send to AI detector (Claude API)
      4. Run full preprocessing pipeline
      5. Store clean DataFrame in session store
      6. Return metadata + preview to frontend

    The frontend shows the preview + detection result and lets the
    user confirm before triggering POST /api/forecast/run.
    """

    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "invalid_file_type",
                "message": "Only CSV files are supported.",
                "field":   "file",
            },
        )

    # ── Read file ─────────────────────────────────────────────────────────────
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "empty_file",
                "message": "The uploaded file is empty.",
                "field":   "file",
            },
        )

    logger.info("Upload received: %s (%d bytes)", file.filename, len(file_bytes))

    # ── Quick parse for column detection ─────────────────────────────────────
    try:
        import io
        import pandas as pd
        raw_df      = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
        columns     = list(raw_df.columns)
        sample_rows = raw_df.to_dict(orient="records")
        total_rows  = sum(1 for _ in io.BytesIO(file_bytes)) - 1  # subtract header
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "parse_error",
                "message": f"Could not read CSV: {exc}",
                "field":   "file",
            },
        )

    # ── AI column detection ───────────────────────────────────────────────────
    try:
        detection = detect_columns(columns, sample_rows)
    except Exception as exc:
        logger.error("Column detection failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code":    "detection_failed",
                "message": f"Column detection failed: {exc}",
                "field":   None,
            },
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    try:
        df, meta = preprocess(
            file_bytes      = file_bytes,
            detected_cols   = detection["detected"],
            detection_mode  = detection["detection_mode"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "preprocessing_failed",
                "message": str(exc),
                "field":   None,
            },
        )

    # ── Store session ─────────────────────────────────────────────────────────
    _purge_expired_sessions()   # housekeeping

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "df":             df,
        "detected_cols":  detection["detected"],
        "detection_mode": detection["detection_mode"],
        "filename":       file.filename,
        "meta":           meta,
        "expires_at":     datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    }

    logger.info(
        "Session created: %s | rows=%d | mode=%s",
        session_id, len(df), detection["detection_mode"],
    )

    # ── Build response ────────────────────────────────────────────────────────
    detected_cols_schema = DetectedColumns(**detection["detected"])

    return UploadResponse(
        session_id       = session_id,
        filename         = file.filename,
        rows             = meta["rows_clean"],
        columns_found    = len(columns),
        detected_columns = detected_cols_schema,
        detection_mode   = DetectionMode(detection["detection_mode"]),
        confidence       = detection["confidence"],
        preview          = sample_rows[:5],
        warnings         = detection["warnings"],
    )
# POST /api/upload — parse CSV, detect columns via Claude AI, preprocess, store session.

from __future__ import annotations

import io
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DetectedColumns, DetectionMode, UploadResponse
from app.services.ai_detector import detect_columns
from app.services.preprocessing import preprocess

logger = logging.getLogger(__name__)
router = APIRouter()

_BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_UPLOADS_DIR = os.path.join(_BACKEND_DIR, "storage", "uploads")
os.makedirs(_UPLOADS_DIR, exist_ok=True)


def _save_upload(session_id: str, filename: str, file_bytes: bytes) -> str:
    # Save raw CSV to storage/uploads/<session_id[:8]>_<filename>.
    safe_name = f"{session_id[:8]}_{filename}"
    filepath  = os.path.join(_UPLOADS_DIR, safe_name)
    with open(filepath, "wb") as f:
        f.write(file_bytes)
    logger.info("CSV saved: %s", filepath)
    return filepath


SESSION_TTL_HOURS = 2

_sessions: dict[str, dict[str, Any]] = {}


def get_session(session_id: str) -> dict[str, Any]:
    # Look up a session and raise 404/410 if missing or expired.
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
    # Remove sessions past their TTL to avoid unbounded memory growth.
    now     = datetime.utcnow()
    expired = [sid for sid, s in _sessions.items() if now > s["expires_at"]]
    for sid in expired:
        _sessions.pop(sid, None)
    if expired:
        logger.info("Purged %d expired sessions", len(expired))


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "invalid_file_type",
                "message": "Only CSV files are supported.",
                "field":   "file",
            },
        )

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

    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes), nrows=10)

        # NSRDB files have a metadata row before the real column headers
        meta_indicators = ["source", "location id", "version", "units"]
        col_lower = [str(c).lower() for c in raw_df.columns]

        if any(ind in col_lower for ind in meta_indicators):
            logger.info("Multi-row header detected — skipping rows 0 and 1, using row 2 as headers")
            raw_df = pd.read_csv(io.BytesIO(file_bytes), skiprows=[0, 1], nrows=5)

        columns     = list(raw_df.columns)
        sample_rows = raw_df.head(5).to_dict(orient="records")

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "parse_error",
                "message": f"Could not read CSV: {exc}",
                "field":   "file",
            },
        )

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

    try:
        df, meta = preprocess(
            file_bytes     = file_bytes,
            detected_cols  = detection["detected"],
            detection_mode = detection["detection_mode"],
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

    _purge_expired_sessions()
    session_id = str(uuid.uuid4())
    filepath   = _save_upload(session_id, file.filename, file_bytes)

    _sessions[session_id] = {
        "df":             df,
        "detected_cols":  detection["detected"],
        "detection_mode": detection["detection_mode"],
        "filename":       file.filename,
        "filepath":       filepath,
        "meta":           meta,
        "expires_at":     datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    }

    try:
        from app.routers.correction import register_upload_session

        register_upload_session(session_id, df, detection["detected"])
    except Exception as exc:
        logger.warning("Failed to register correction session for %s: %s", session_id, exc)

    logger.info(
        "Session created: %s | rows=%d | mode=%s",
        session_id, len(df), detection["detection_mode"],
    )

    detected_cols_schema = DetectedColumns(**detection["detected"])

    return UploadResponse(
        session_id       = session_id,
        filename         = file.filename,
        rows             = meta["rows_clean"],
        columns_found    = len(columns),
        detected_columns = detected_cols_schema,
        detection_mode   = DetectionMode(detection["detection_mode"]),
        confidence       = detection["confidence"],
        data_stats       = meta,
        preview          = sample_rows[:5],
        warnings         = detection["warnings"],
    )

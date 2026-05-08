# POST /api/upload — parse CSV, detect columns via Claude AI, preprocess, store session.

from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import pickle
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db, save_dataset
from app.models.schemas import DetectedColumns, DetectionMode, UploadResponse
from app.services.ai_detector import detect_columns
from ml_core.preprocessing import preprocess

logger = logging.getLogger(__name__)
router = APIRouter()

_BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SESSIONS_DIR = os.path.join(_BACKEND_DIR, "storage", "sessions")
os.makedirs(_SESSIONS_DIR, exist_ok=True)


SESSION_TTL_HOURS = 2


def _session_path(session_id: str) -> str:
    return os.path.join(_SESSIONS_DIR, f"{session_id}.pkl")


def _write_session(session_id: str, data: dict[str, Any]) -> None:
    with open(_session_path(session_id), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_session(session_id: str) -> dict[str, Any] | None:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load session %s: %s", session_id, exc)
        return None


def _delete_session(session_id: str) -> None:
    """Remove the session pickle and any exports tied to it."""
    try:
        os.remove(_session_path(session_id))
    except FileNotFoundError:
        pass

    _EXPORTS_DIR = os.path.join(_BACKEND_DIR, "storage", "exports")
    if os.path.isdir(_EXPORTS_DIR):
        export_prefix = f"solar_forecast_{session_id[:8]}_"
        for fname in os.listdir(_EXPORTS_DIR):
            if fname.startswith(export_prefix):
                try:
                    os.remove(os.path.join(_EXPORTS_DIR, fname))
                    logger.info("Deleted export: %s", fname)
                except OSError:
                    pass


def get_session(session_id: str) -> dict[str, Any]:
    # Look up a session and raise 404/410 if missing or expired.
    session = _read_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail={
                "code":    "session_not_found",
                "message": "Session not found. Please upload a file first.",
                "field":   "session_id",
            },
        )

    if datetime.now(timezone.utc).replace(tzinfo=None) > session["expires_at"]:
        _delete_session(session_id)
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
    # Remove all artifacts (session, upload, exports) for sessions past their TTL.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        for fname in os.listdir(_SESSIONS_DIR):
            if not fname.endswith(".pkl"):
                continue
            path = os.path.join(_SESSIONS_DIR, fname)
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if now > data.get("expires_at", now):
                    session_id = fname[:-4]  # strip .pkl
                    _delete_session(session_id)
                    logger.info("Purged expired session and artifacts: %s", session_id[:8])
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Session purge failed: %s", exc)


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "invalid_file_type",
                "message": "Please upload a CSV file.",
                "field":   "file",
            },
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "empty_file",
                "message": "The uploaded file is empty. Please check the file and try again.",
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
                "message": f"We couldn't read the CSV file — {exc}",
                "field":   "file",
            },
        )

    try:
        detection = detect_columns(columns, sample_rows)
    except ValueError as exc:
        # Irrelevant dataset — no solar or weather columns found at all
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "irrelevant_dataset",
                "message": str(exc),
                "field":   None,
            },
        )
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
    file_hash  = hashlib.sha256(file_bytes).hexdigest()

    # Strip None values from col_map — pipeline's add_weather_features skips missing keys,
    # but a key present with None value causes it to look up df[None] and crash.
    clean_col_map = {k: v for k, v in detection["detected"].items() if v is not None}

    session_data = {
        "df":             df,
        "detected_cols":  clean_col_map,
        "detection_mode": detection["detection_mode"],
        "filename":       file.filename,
        "file_hash":      file_hash,
        "meta":           meta,
        "expires_at":     datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=SESSION_TTL_HOURS),
    }

    try:
        dataset = save_dataset(
            db,
            session_id     = session_id,
            filename       = file.filename,
            file_path      = "",
            file_size      = len(file_bytes),
            file_hash      = file_hash,
            row_count      = meta["rows_clean"],
            column_map     = json.dumps(detection["detected"]),
            detection_mode = detection["detection_mode"],
        )
        session_data["dataset_id"] = dataset.dataset_id
    except Exception as exc:
        logger.warning("Failed to persist dataset metadata for %s: %s", session_id, exc)

    _write_session(session_id, session_data)

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

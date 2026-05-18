# FastAPI router for anomaly detection.

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.routers.upload import get_session
from ml_core.anomaly import detect_anomalies

logger = logging.getLogger(__name__)
router = APIRouter()

# Return detected anomalies for an uploaded session.
@router.get("/anomaly")
def get_anomalies(session_id: str):
    session = get_session(session_id)
    df = session["df"]

    try:
        result = detect_anomalies(df)
    except Exception as exc:
        logger.error("Anomaly detection failed for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "anomaly_detection_failed",
                "message": f"Anomaly detection failed: {exc}",
                "field": None,
            },
        )

    result["session_id"] = session_id
    return result
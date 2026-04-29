# GET /api/anomaly — run anomaly detection on a previously uploaded dataset.

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import AnomalyRecord, AnomalyResponse
from app.routers.upload import get_session
from app.services.anomaly import detect_anomalies

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/anomaly", response_model=AnomalyResponse)
def get_anomalies(session_id: str):
    # Run Z-score + IQR + rolling window detection, merge results, and return them.
    session = get_session(session_id)
    df      = session["df"]

    logger.info("Anomaly detection requested | session=%s", session_id)

    try:
        result = detect_anomalies(df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code":    "anomaly_detection_failed",
                "message": str(exc),
                "field":   None,
            },
        )

    anomaly_records = [
        AnomalyRecord(
            timestamp = a["timestamp"],
            value     = a["value"],
            expected  = a["expected"],
            deviation = a["deviation"],
            severity  = a["severity"],
            method    = a["method"],
        )
        for a in result["anomalies"]
    ]

    return AnomalyResponse(
        session_id    = session_id,
        total_points  = result["total_points"],
        anomaly_count = result["anomaly_count"],
        anomaly_rate  = result["anomaly_rate"],
        anomalies     = anomaly_records,
    )

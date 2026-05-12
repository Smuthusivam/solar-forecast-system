# GET /api/export/csv         — download the preprocessed dataset as CSV.
# GET /api/export/features    — download the feature-engineered dataset as CSV.

from __future__ import annotations

import io
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session
from ml_core.feature_engineering import build_features

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/export/csv")
def export_csv(session_id: str):
    session  = get_session(session_id)
    df       = session["df"]
    filename = session["filename"]

    # Reset index so the timestamp column is included in the CSV
    export_df = df.reset_index()

    # Ensure timestamp is formatted as a clean ISO string without timezone offsets
    if "timestamp" in export_df.columns:
        export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Put timestamp first
    cols = ["timestamp"] + [c for c in export_df.columns if c != "timestamp"]
    export_df = export_df[cols]

    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode()

    stem        = filename.rsplit(".", 1)[0] if "." in filename else filename
    export_name = f"{stem}_preprocessed.csv"

    logger.info("Exporting preprocessed data: %s (%d rows)", export_name, len(df))

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={export_name}"},
    )


@router.get("/export/features")
def export_features(session_id: str):
    session  = get_session(session_id)
    df       = session["df"]
    col_map  = session.get("detected_cols", {})
    filename = session["filename"]

    feat_df = build_features(df, target_col="irradiance", col_map=col_map)
    export_df = feat_df.reset_index()

    if "timestamp" in export_df.columns:
        export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["timestamp"] + [c for c in export_df.columns if c != "timestamp"]
    export_df = export_df[cols]

    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode()

    stem        = filename.rsplit(".", 1)[0] if "." in filename else filename
    export_name = f"{stem}_features.csv"

    logger.info("Exporting feature-engineered data: %s (%d rows, %d cols)", export_name, len(export_df), len(export_df.columns))

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={export_name}"},
    )

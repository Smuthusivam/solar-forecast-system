# GET /api/export/csv — download the preprocessed dataset as CSV.

from __future__ import annotations

import io
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/export/csv")
def export_csv(session_id: str):
    session  = get_session(session_id)
    df       = session["df"]
    filename = session["filename"]

    # Reset index so the timestamp column is included in the CSV
    export_df = df.reset_index()

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

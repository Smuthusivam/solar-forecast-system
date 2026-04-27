"""
export.py — GET /api/export/csv
            GET /api/export/pdf
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session
from app.services.anomaly import detect_anomalies
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/export/csv")
def export_csv(session_id: str, horizon: int = 24):
    """
    Run pipeline and return forecast results as a downloadable CSV file.
    """
    session = get_session(session_id)
    df      = session["df"]
    col_map = session["detected_cols"]

    try:
        result = run_pipeline(df, col_map, horizon=horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Build CSV from forecast points
    import pandas as pd
    rows = [
        {
            "timestamp": p["timestamp"],
            "predicted": p["predicted"],
            "actual":    p["actual"],
            "lower":     p["lower"],
            "upper":     p["upper"],
        }
        for p in result["forecast"]
    ]
    export_df = pd.DataFrame(rows)

    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"solar_forecast_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode()),
        media_type = "text/csv",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/pdf")
def export_pdf(session_id: str, horizon: int = 24):
    """
    Generate a PDF report summarising the forecast run.
    Uses reportlab if available, falls back to plain text PDF.
    """
    session = get_session(session_id)
    df      = session["df"]
    col_map = session["detected_cols"]
    filename = session["filename"]

    try:
        result   = run_pipeline(df, col_map, horizon=horizon)
        anomalies = detect_anomalies(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []

        em = result["ensemble_metrics"]

        # Title
        story.append(Paragraph("Solar Irradiance Forecast Report", styles["Title"]))
        story.append(Spacer(1, 12))

        # Summary
        story.append(Paragraph(f"File: {filename}", styles["Normal"]))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
        story.append(Paragraph(f"Horizon: {horizon} hours", styles["Normal"]))
        story.append(Paragraph(f"Detection mode: {result['detection_mode']}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Ensemble metrics table
        story.append(Paragraph("Ensemble Model Performance", styles["Heading2"]))
        metrics_data = [
            ["Metric", "Value"],
            ["RMSE",   f"{em['rmse']:.2f} W/m²"],
            ["MAE",    f"{em['mae']:.2f} W/m²"],
            ["R²",     f"{em['r2']:.4f}"],
            ["MAPE",   f"{em['mape']:.2f}%"],
        ]
        t = Table(metrics_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("PADDING",    (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Per-model table
        story.append(Paragraph("Individual Model Results", styles["Heading2"]))
        model_data = [["Model", "RMSE", "MAE", "R²", "Weight"]]
        for m in result["per_model"]:
            model_data.append([
                m["model_name"],
                f"{m['metrics']['rmse']:.2f}",
                f"{m['metrics']['mae']:.2f}",
                f"{m['metrics']['r2']:.4f}",
                f"{m['weight']:.3f}",
            ])
        t2 = Table(model_data, colWidths=[120, 80, 80, 80, 80])
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("PADDING",       (0, 0), (-1, -1), 8),
        ]))
        story.append(t2)
        story.append(Spacer(1, 12))

        # Anomaly summary
        story.append(Paragraph("Anomaly Detection Summary", styles["Heading2"]))
        story.append(Paragraph(
            f"Total anomalies detected: {anomalies['anomaly_count']} "
            f"({anomalies['anomaly_rate']*100:.1f}% of data points)",
            styles["Normal"],
        ))

        doc.build(story)
        buffer.seek(0)

        pdf_filename = f"solar_forecast_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        return StreamingResponse(
            buffer,
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f"attachment; filename={pdf_filename}"},
        )

    except ImportError:
        # reportlab not installed — return plain text fallback
        logger.warning("reportlab not installed — returning text report")
        em = result["ensemble_metrics"]
        text = f"""SOLAR IRRADIANCE FORECAST REPORT
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
File: {filename}
Horizon: {horizon} hours

ENSEMBLE METRICS
RMSE : {em['rmse']:.2f} W/m²
MAE  : {em['mae']:.2f} W/m²
R²   : {em['r2']:.4f}
MAPE : {em['mape']:.2f}%

ANOMALIES
Total: {anomalies['anomaly_count']} ({anomalies['anomaly_rate']*100:.1f}%)
"""
        return StreamingResponse(
            io.BytesIO(text.encode()),
            media_type = "text/plain",
            headers    = {"Content-Disposition": f"attachment; filename=report.txt"},
        )
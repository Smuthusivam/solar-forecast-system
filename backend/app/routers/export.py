# GET /api/export/csv and GET /api/export/pdf — download forecast results.

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.routers.upload import get_session
from ml_core.anomaly import detect_anomalies
from ml_core.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

_BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_EXPORTS_DIR = os.path.join(_BACKEND_DIR, "storage", "exports")
os.makedirs(_EXPORTS_DIR, exist_ok=True)


def _save_export(filename: str, data: bytes) -> str:
    # Save the export file to storage/exports/ and return its path.
    filepath = os.path.join(_EXPORTS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    logger.info("Export saved: %s", filepath)
    return filepath


@router.get("/export/csv")
def export_csv(session_id: str, horizon: int = 24):
    # Run the pipeline and return forecast results as a downloadable CSV.
    session = get_session(session_id)
    df      = session["df"]
    col_map = session["detected_cols"]

    try:
        result = run_pipeline(df, col_map, horizon=horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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
    csv_bytes = buffer.getvalue().encode()

    filename = f"solar_forecast_{session_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    _save_export(filename, csv_bytes)

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type = "text/csv",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/pdf")
def export_pdf(session_id: str, horizon: int = 24):
    # Generate a PDF report; falls back to plain text if ReportLab is not installed.
    session  = get_session(session_id)
    df       = session["df"]
    col_map  = session["detected_cols"]
    filename = session["filename"]

    try:
        result    = run_pipeline(df, col_map, horizon=horizon)
        anomalies = detect_anomalies(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    em           = result["metrics"]
    pdf_filename = f"solar_forecast_{session_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        )

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []

        story.append(Paragraph("Solar Irradiance Forecast Report", styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"<b>File:</b> {filename}",                                          styles["Normal"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Horizon:</b> {horizon} hours",                                  styles["Normal"]))
        story.append(Paragraph(f"<b>Detection mode:</b> {result['detection_mode']}",                styles["Normal"]))
        story.append(Paragraph(f"<b>Rows processed:</b> {result['rows_processed']}",                styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"Model Performance ({result['best_model']})", styles["Heading2"]))
        metrics_data = [
            ["Metric", "Value"],
            ["RMSE",   f"{em['rmse']:.2f} W/m²"],
            ["MAE",    f"{em['mae']:.2f} W/m²"],
            ["R²",     f"{em['r2']:.4f}"],
        ]
        t = Table(metrics_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
            ("PADDING",        (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Individual Model Results", styles["Heading2"]))
        model_data = [["Model", "RMSE", "MAE", "R²", "Best"]]
        for mi in result["models_info"]:
            model_data.append([
                mi["model_name"],
                f"{mi['metrics']['rmse']:.2f}",
                f"{mi['metrics']['mae']:.2f}",
                f"{mi['metrics']['r2']:.4f}",
                "Yes" if mi["is_best"] else "No",
            ])
        t2 = Table(model_data, colWidths=[120, 80, 80, 80, 80])
        t2.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
            ("PADDING",        (0, 0), (-1, -1), 8),
        ]))
        story.append(t2)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Anomaly Detection Summary", styles["Heading2"]))
        story.append(Paragraph(
            f"Total anomalies detected: {anomalies['anomaly_count']} "
            f"({anomalies['anomaly_rate'] * 100:.1f}% of data points)",
            styles["Normal"],
        ))

        if anomalies["anomalies"]:
            story.append(Spacer(1, 8))
            anom_data = [["Timestamp", "Value (W/m²)", "Expected", "Severity", "Method"]]
            for a in anomalies["anomalies"][:10]:
                anom_data.append([
                    a["timestamp"][:16],
                    f"{a['value']:.1f}",
                    f"{a['expected']:.1f}",
                    a["severity"],
                    a["method"],
                ])
            t3 = Table(anom_data, colWidths=[130, 80, 80, 70, 70])
            t3.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
                ("PADDING",        (0, 0), (-1, -1), 6),
            ]))
            story.append(t3)

        doc.build(story)
        pdf_bytes = buffer.getvalue()

        _save_export(pdf_filename, pdf_bytes)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f"attachment; filename={pdf_filename}"},
        )

    except ImportError:
        # reportlab not installed — return a plain text report instead
        logger.warning("reportlab not installed — returning plain text report")

        text = (
            f"SOLAR IRRADIANCE FORECAST REPORT\n"
            f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"File      : {filename}\n"
            f"Horizon   : {horizon} hours\n"
            f"Mode      : {result['detection_mode']}\n"
            f"Model     : {result['best_model']}\n\n"
            f"MODEL METRICS\n"
            f"RMSE : {em['rmse']:.2f} W/m²\n"
            f"MAE  : {em['mae']:.2f} W/m²\n"
            f"R²   : {em['r2']:.4f}\n\n"
            f"ANOMALIES\n"
            f"Total: {anomalies['anomaly_count']} "
            f"({anomalies['anomaly_rate'] * 100:.1f}%)\n"
        )
        txt_bytes    = text.encode()
        txt_filename = pdf_filename.replace(".pdf", ".txt")

        _save_export(txt_filename, txt_bytes)

        return StreamingResponse(
            io.BytesIO(txt_bytes),
            media_type = "text/plain",
            headers    = {"Content-Disposition": f"attachment; filename={txt_filename}"},
        )

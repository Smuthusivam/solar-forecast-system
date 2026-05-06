import csv
import io
import os
import sys
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _build_csv_bytes(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp",
        "irradiance",
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
    ])

    start = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(rows):
        ts = start + timedelta(hours=i)
        hour = ts.hour
        irradiance = max(0.0, 500.0 * (1 - abs(12 - hour) / 12))
        temperature = 20 + (hour - 12) * 0.5
        humidity = 60 - (hour - 12) * 1.2
        wind_speed = 3.5 + (hour % 4) * 0.3
        cloud_cover = 30 + (hour % 5) * 5
        writer.writerow([
            ts.isoformat(),
            round(irradiance, 2),
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(cloud_cover, 2),
        ])

    return output.getvalue().encode()


def _build_csv_with_metadata(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Source", "Location ID", "Version", "Units", "Extra1", "Extra2"])
    writer.writerow(["NSRDB", "123", "3.0", "metadata", "-", "-"])
    writer.writerow([
        "timestamp",
        "irradiance",
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
    ])

    start = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(rows):
        ts = start + timedelta(hours=i)
        hour = ts.hour
        irradiance = max(0.0, 520.0 * (1 - abs(12 - hour) / 12))
        temperature = 22 + (hour - 12) * 0.4
        humidity = 55 - (hour - 12) * 1.0
        wind_speed = 4.0 + (hour % 3) * 0.2
        cloud_cover = 25 + (hour % 6) * 4
        writer.writerow([
            ts.isoformat(),
            round(irradiance, 2),
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(cloud_cover, 2),
        ])

    return output.getvalue().encode()


def _build_csv_nsrdb(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Year",
        "Month",
        "Day",
        "Hour",
        "GHI",
        "Temperature",
        "Relative Humidity",
        "Wind Speed",
        "Cloud Cover",
    ])

    start = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(rows):
        ts = start + timedelta(hours=i)
        hour = ts.hour
        irradiance = max(0.0, 480.0 * (1 - abs(12 - hour) / 12))
        temperature = 21 + (hour - 12) * 0.5
        humidity = 65 - (hour - 12) * 0.9
        wind_speed = 3.0 + (hour % 5) * 0.2
        cloud_cover = 35 + (hour % 4) * 5
        writer.writerow([
            ts.year,
            ts.month,
            ts.day,
            ts.hour,
            round(irradiance, 2),
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(cloud_cover, 2),
        ])

    return output.getvalue().encode()


def _build_csv_no_timestamp(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "irradiance",
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
    ])

    for i in range(rows):
        hour = i % 24
        irradiance = max(0.0, 510.0 * (1 - abs(12 - hour) / 12))
        temperature = 20 + (hour - 12) * 0.4
        humidity = 60 - (hour - 12) * 1.1
        wind_speed = 3.2 + (hour % 4) * 0.25
        cloud_cover = 28 + (hour % 5) * 6
        writer.writerow([
            round(irradiance, 2),
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(cloud_cover, 2),
        ])

    return output.getvalue().encode()


def _build_csv_no_irradiance(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
    ])

    start = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(rows):
        ts = start + timedelta(hours=i)
        hour = ts.hour
        temperature = 19 + (hour - 12) * 0.4
        humidity = 62 - (hour - 12) * 0.8
        cloud_cover = 40 + (hour % 6) * 6
        wind_speed = 3.0 + (hour % 4) * 0.3
        writer.writerow([
            ts.isoformat(),
            round(temperature, 2),
            round(humidity, 2),
            round(cloud_cover, 2),
            round(wind_speed, 2),
        ])

    return output.getvalue().encode()


def _build_csv_bad_timestamp(rows: int = 72) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp",
        "irradiance",
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
    ])

    for i in range(rows):
        hour = i % 24
        irradiance = max(0.0, 500.0 * (1 - abs(12 - hour) / 12))
        temperature = 20 + (hour - 12) * 0.4
        humidity = 60 - (hour - 12) * 1.1
        wind_speed = 3.2 + (hour % 4) * 0.25
        cloud_cover = 28 + (hour % 5) * 6
        writer.writerow([
            "not-a-timestamp",
            round(irradiance, 2),
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(cloud_cover, 2),
        ])

    return output.getvalue().encode()


def _stub_detect_columns(columns, sample_rows):
    return {
        "detected": {
            "irradiance": "irradiance",
            "temperature": "temperature",
            "humidity": "humidity",
            "wind_speed": "wind_speed",
            "cloud_cover": "cloud_cover",
            "sunshine_hours": None,
            "timestamp": "timestamp",
        },
        "detection_mode": "direct",
        "confidence": 0.99,
        "warnings": [],
        "used_fallback": True,
    }


def test_backend_smoke(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router
    from app import database as db_module

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code in (200, 503)
    assert "status" in health.json()

    root = client.get("/")
    assert root.status_code == 200
    assert "message" in root.json()

    csv_bytes = _build_csv_bytes()
    upload = client.post(
        "/api/upload",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert upload.status_code == 200
    upload_payload = upload.json()
    session_id = upload_payload["session_id"]

    forecast = client.post(
        "/api/forecast/run",
        json={
            "session_id": session_id,
            "horizon": 24,
            "train_size": 80,
        },
    )
    assert forecast.status_code == 200
    forecast_payload = forecast.json()
    assert "forecast" in forecast_payload
    assert "per_model" in forecast_payload
    assert forecast_payload["run_id"] > 0

    with db_module.SessionLocal() as db:
        run = db_module.get_run_by_id(db, forecast_payload["run_id"])
        assert run is not None
        assert run.session_id == session_id

    anomaly = client.get(f"/api/anomaly?session_id={session_id}")
    assert anomaly.status_code == 200
    anomaly_payload = anomaly.json()
    assert "anomaly_count" in anomaly_payload

    export_csv = client.get(f"/api/export/csv?session_id={session_id}&horizon=24")
    assert export_csv.status_code == 200
    assert export_csv.headers.get("content-type", "").startswith("text/csv")

    export_pdf = client.get(f"/api/export/pdf?session_id={session_id}&horizon=24")
    assert export_pdf.status_code == 200
    assert "content-disposition" in export_pdf.headers

    history = client.get("/api/history")
    assert history.status_code == 200
    history_payload = history.json()
    assert "runs" in history_payload


def test_correction_flow(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router
    from app.routers import correction as correction_router
    from app.services import anomaly as anomaly_service
    from app import services as services_pkg

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    def _stub_detect_anomalies(_df):
        return {
            "anomaly_count": 1,
            "anomaly_rate": 0.01,
            "total_points": 100,
            "anomalies": [
                {
                    "timestamp": "2024-01-01T05:00:00",
                    "value": 900.0,
                    "expected": 450.0,
                    "deviation": 450.0,
                    "severity": "high",
                    "method": "zscore",
                }
            ],
        }

    async def _stub_correct_all_async(df, _anomalies, _column_map):
        return df.copy(), [
            {
                "timestamp": "2024-01-01T05:00:00",
                "original_value": 900.0,
                "corrected_value": 450.0,
                "reasoning": "Corrected spike to daytime trend.",
                "confidence": "high",
                "method_flagged_by": "zscore",
                "severity": "high",
                "correction_source": "ai",
            }
        ]

    def _stub_correction_stats(_log):
        return {
            "total_corrected": 1,
            "ai_corrections": 1,
            "physics_rule_corrections": 0,
            "interpolation_fallbacks": 0,
            "avg_correction_delta": 450.0,
            "max_correction_delta": 450.0,
            "high_confidence": 1,
            "medium_confidence": 0,
            "low_confidence": 0,
        }

    def _stub_run_pipeline(_df, _col_map, horizon=24, detection_mode="direct", train_size=80):
        return {
            "forecast": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "predicted": 100.0,
                    "actual": 120.0,
                    "lower": None,
                    "upper": None,
                }
            ],
            "future_forecast": [],
            "ensemble_metrics": {"rmse": 10.0, "mae": 8.0, "r2": 0.9, "mape": 5.0},
            "per_model": [
                {
                    "model_name": "XGBoost",
                    "predictions": [100.0],
                    "metrics": {"rmse": 12.0, "mae": 9.0, "r2": 0.88, "mape": 6.0},
                    "train_metrics": {"rmse": 11.0, "mae": 8.0, "r2": 0.9, "mape": 5.0},
                    "weight": 0.6,
                },
                {
                    "model_name": "LightGBM",
                    "predictions": [100.0],
                    "metrics": {"rmse": 11.0, "mae": 8.5, "r2": 0.89, "mape": 5.5},
                    "train_metrics": {"rmse": 10.5, "mae": 8.0, "r2": 0.91, "mape": 5.0},
                    "weight": 0.4,
                },
            ],
            "feature_importance": {},
            "detection_mode": detection_mode,
            "horizon": horizon,
            "rows_processed": 100,
        }

    monkeypatch.setattr(anomaly_service, "detect_anomalies", _stub_detect_anomalies)
    monkeypatch.setattr(correction_router, "detect_anomalies", _stub_detect_anomalies)
    monkeypatch.setattr(correction_router, "_correct_all_async", _stub_correct_all_async)
    monkeypatch.setattr(correction_router, "compute_correction_stats", _stub_correction_stats)
    monkeypatch.setattr(services_pkg.pipeline, "run_pipeline", _stub_run_pipeline)

    client = TestClient(app)
    csv_bytes = _build_csv_bytes()
    upload = client.post(
        "/api/upload",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    correction = client.post(f"/api/correction/run/{session_id}")
    assert correction.status_code == 200
    correction_payload = correction.json()
    assert correction_payload["anomaly_count"] == 1
    assert correction_payload["anomalies_corrected"] == 1
    assert "metrics_comparison" in correction_payload
    assert correction_payload["session_id"]

    export_corrected = client.get(
        f"/api/correction/export/{correction_payload['session_id']}"
    )
    assert export_corrected.status_code == 200
    assert export_corrected.headers.get("content-type", "").startswith("text/csv")

    log = client.get(
        f"/api/correction/log/{correction_payload['session_id']}"
    )
    assert log.status_code == 200
    assert log.json()["total"] == 1


@pytest.mark.parametrize(
    "payload, detected_cols, detection_mode",
    [
        (
            _build_csv_with_metadata(),
            {
                "irradiance": "irradiance",
                "temperature": "temperature",
                "humidity": "humidity",
                "wind_speed": "wind_speed",
                "cloud_cover": "cloud_cover",
                "sunshine_hours": None,
                "timestamp": "timestamp",
            },
            "direct",
        ),
        (
            _build_csv_nsrdb(),
            {
                "irradiance": "GHI",
                "temperature": "Temperature",
                "humidity": "Relative Humidity",
                "wind_speed": "Wind Speed",
                "cloud_cover": "Cloud Cover",
                "sunshine_hours": None,
                "timestamp": None,
            },
            "direct",
        ),
        (
            _build_csv_no_timestamp(),
            {
                "irradiance": "irradiance",
                "temperature": "temperature",
                "humidity": "humidity",
                "wind_speed": "wind_speed",
                "cloud_cover": "cloud_cover",
                "sunshine_hours": None,
                "timestamp": None,
            },
            "direct",
        ),
    ],
)
def test_upload_variants(monkeypatch, payload, detected_cols, detection_mode):
    from app.main import app
    from app.routers import upload as upload_router

    def _stub_detect_columns_variant(_columns, _sample_rows):
        return {
            "detected": detected_cols,
            "detection_mode": detection_mode,
            "confidence": 0.95,
            "warnings": [],
            "used_fallback": True,
        }

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns_variant)

    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] >= 48
    assert body["session_id"]


def test_upload_estimated_ghi(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    def _stub_detect_columns_variant(_columns, _sample_rows):
        return {
            "detected": {
                "irradiance": None,
                "temperature": "temperature",
                "humidity": "humidity",
                "wind_speed": "wind_speed",
                "cloud_cover": "cloud_cover",
                "sunshine_hours": None,
                "timestamp": "timestamp",
            },
            "detection_mode": "estimated",
            "confidence": 0.92,
            "warnings": [],
            "used_fallback": True,
        }

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns_variant)

    client = TestClient(app)
    payload = _build_csv_no_irradiance()
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detection_mode"] == "estimated"
    assert body["rows"] >= 48


def test_upload_rejects_empty_file(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_file"


def test_upload_rejects_non_csv(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("sample.txt", _build_csv_bytes(), "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_file_type"


def test_upload_rejects_too_few_rows(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)
    payload = _build_csv_bytes(rows=12)
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert response.status_code == 422


def test_upload_accepts_bad_timestamp_with_fallback(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)
    payload = _build_csv_bad_timestamp()
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["rows"] >= 48


def test_anomaly_missing_session():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/anomaly?session_id=missing-session")
    assert response.status_code == 404


def test_export_missing_session():
    from app.main import app

    client = TestClient(app)
    csv_response = client.get("/api/export/csv?session_id=missing-session&horizon=24")
    pdf_response = client.get("/api/export/pdf?session_id=missing-session&horizon=24")
    assert csv_response.status_code == 404
    assert pdf_response.status_code == 404


def test_forecast_invalid_payload(monkeypatch):
    from app.main import app
    from app.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "detect_columns", _stub_detect_columns)

    client = TestClient(app)
    payload = _build_csv_bytes()
    upload = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    bad_horizon = client.post(
        "/api/forecast/run",
        json={"session_id": session_id, "horizon": 0, "train_size": 80},
    )
    assert bad_horizon.status_code == 422

    bad_train = client.post(
        "/api/forecast/run",
        json={"session_id": session_id, "horizon": 24, "train_size": 10},
    )
    assert bad_train.status_code == 422

    missing_session = client.post(
        "/api/forecast/run",
        json={"horizon": 24, "train_size": 80},
    )
    assert missing_session.status_code == 422

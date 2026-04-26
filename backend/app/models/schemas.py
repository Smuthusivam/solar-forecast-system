"""
Pydantic v2 schemas for the Solar Irradiance Forecasting System.

Design rules:
  - All response models use model_config with from_attributes=True so
    SQLAlchemy ORM objects can be serialised directly.
  - Optional fields that may not be present (e.g. weather columns the
    user didn't supply) default to None — never raise on missing keys.
  - Numeric fields use float, not int, because irradiance values are
    continuous (W/m²).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────
# Shared enums
# ─────────────────────────────────────────────

class DetectionMode(str, Enum):
    """
    direct   → a GHI/irradiance column was found in the CSV.
    estimated → no irradiance column; GHI will be estimated from
                weather variables via the Angstrom-Prescott formula.
    """
    DIRECT = "direct"
    ESTIMATED = "estimated"


class ForecastHorizon(int, Enum):
    H24 = 24
    H48 = 48
    H72 = 72


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnomalyMethod(str, Enum):
    ZSCORE = "zscore"
    IQR = "iqr"
    ROLLING = "rolling"


# ─────────────────────────────────────────────
# Upload domain
# ─────────────────────────────────────────────

class DetectedColumns(BaseModel):
    """
    Mapping returned by the Claude API detector.
    Every field is Optional because a given CSV may or may not contain
    each physical variable.  The only guarantee is that after preprocessing
    we will always have an irradiance series (real or estimated).
    """
    irradiance:   Optional[str] = Field(None, description="GHI column name in the raw CSV")
    temperature:  Optional[str] = None
    humidity:     Optional[str] = None
    wind_speed:   Optional[str] = None
    cloud_cover:  Optional[str] = None
    sunshine_hours: Optional[str] = None
    timestamp:    Optional[str] = Field(None, description="Datetime column name")


class UploadResponse(BaseModel):
    """
    Returned immediately after the user uploads a CSV.
    The frontend uses `session_id` as the key for all subsequent calls.
    """
    session_id:      str   = Field(..., description="UUID that identifies this upload session")
    filename:        str
    rows:            int   = Field(..., description="Number of data rows detected (excl. header)")
    columns_found:   int   = Field(..., description="Total columns in the raw file")
    detected_columns: DetectedColumns
    detection_mode:  DetectionMode
    confidence:      float = Field(..., ge=0.0, le=1.0,
                                   description="Claude API confidence score (0–1)")
    preview:         list[dict] = Field(
        default_factory=list,
        description="First 5 rows of the raw CSV as a list of dicts (column → value)"
    )
    warnings:        list[str] = Field(
        default_factory=list,
        description="Non-fatal issues found during column detection (e.g. missing weather cols)"
    )


# ─────────────────────────────────────────────
# Forecast domain
# ─────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """
    Body sent by the frontend to POST /api/forecast/run.
    """
    session_id: str
    horizon:    ForecastHorizon = ForecastHorizon.H24


class ModelMetrics(BaseModel):
    """
    Evaluation metrics for a single model on the held-out test set.
    All values are floats; None means the metric could not be computed
    (e.g. MAPE is undefined when actuals contain zeros).
    """
    rmse: float = Field(..., description="Root Mean Square Error  (W/m²)")
    mae:  float = Field(..., description="Mean Absolute Error     (W/m²)")
    r2:   float = Field(..., description="Coefficient of Determination (higher = better)")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)")


class ForecastPoint(BaseModel):
    """
    A single time-step in a forecast series.
    `actual` is None for future (out-of-sample) predictions.
    `lower` / `upper` come from Prophet's uncertainty intervals;
    for XGBoost and LightGBM they are also None.
    """
    timestamp: datetime
    predicted: float = Field(..., description="Ensemble-weighted prediction  (W/m²)")
    actual:    Optional[float] = Field(None, description="Ground-truth value if available")
    lower:     Optional[float] = Field(None, description="Lower confidence bound (Prophet)")
    upper:     Optional[float] = Field(None, description="Upper confidence bound (Prophet)")


class PerModelForecast(BaseModel):
    """
    Raw predictions from one individual model — used in the Model
    Comparison page so the user can see how each model performed.
    """
    model_name: str
    predictions: list[float]
    metrics:     ModelMetrics
    weight:      float = Field(..., ge=0.0, le=1.0,
                               description="Ensemble weight assigned to this model")


class ForecastResponse(BaseModel):
    """
    Full response from POST /api/forecast/run.
    """
    session_id:     str
    run_id:         int  = Field(..., description="SQLite primary key for this run (used in /history)")
    horizon:        int  = Field(..., description="Forecast horizon in hours")
    detection_mode: DetectionMode

    # Ensemble result (what the dashboard chart shows)
    forecast:       list[ForecastPoint]
    ensemble_metrics: ModelMetrics

    # Individual model details (what the Model Comparison page shows)
    per_model:      list[PerModelForecast]

    # Feature importances from tree models (column_name → importance score)
    feature_importance: Optional[dict[str, float]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Anomaly domain
# ─────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    """
    One detected anomaly in the irradiance series.
    """
    timestamp:  datetime
    value:      float  = Field(..., description="Observed irradiance value  (W/m²)")
    expected:   float  = Field(..., description="Expected / rolling-mean value")
    deviation:  float  = Field(..., description="Absolute deviation from expected")
    severity:   AnomalySeverity
    method:     AnomalyMethod = Field(..., description="Detection method that flagged this point")


class AnomalyResponse(BaseModel):
    session_id:     str
    total_points:   int   = Field(..., description="Total data points scanned")
    anomaly_count:  int
    anomaly_rate:   float = Field(..., description="anomaly_count / total_points")
    anomalies:      list[AnomalyRecord]


# ─────────────────────────────────────────────
# History domain
# ─────────────────────────────────────────────

class ForecastRunSummary(BaseModel):
    """
    One row in the /history list — lightweight metadata only.
    The full ForecastResponse is not stored in SQLite (too large);
    the user must re-run to get full results.
    """
    model_config = {"from_attributes": True}   # allows ORM → Pydantic conversion

    run_id:         int
    session_id:     str
    filename:       str
    horizon:        int
    detection_mode: DetectionMode
    ensemble_rmse:  float
    ensemble_mae:   float
    ensemble_r2:    float
    rows_processed: int
    anomaly_count:  int
    created_at:     datetime


class HistoryResponse(BaseModel):
    total_runs: int
    runs:       list[ForecastRunSummary]


# ─────────────────────────────────────────────
# Error / health
# ─────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """
    Consistent error envelope returned by all endpoints on failure.
    FastAPI's default HTTPException only returns {"detail": "..."};
    this adds a machine-readable `code` for the frontend to switch on.
    """
    code:    str  = Field(..., description="Snake-case error code, e.g. 'column_not_found'")
    message: str
    field:   Optional[str] = Field(None, description="Which field caused the error, if applicable")


class HealthResponse(BaseModel):
    status:    str = "ok"
    version:   str = "1.0.0"
    db_online: bool
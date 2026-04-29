# Pydantic v2 schemas for all API request and response types.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DetectionMode(str, Enum):
    # "direct" = irradiance column found; "estimated" = GHI derived from weather vars.
    DIRECT    = "direct"
    ESTIMATED = "estimated"


class ForecastHorizon(int, Enum):
    H24 = 24
    H48 = 48
    H72 = 72


class AnomalySeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AnomalyMethod(str, Enum):
    ZSCORE  = "zscore"
    IQR     = "iqr"
    ROLLING = "rolling"


class DetectedColumns(BaseModel):
    # Every field is Optional because a given CSV may not have all physical variables.
    irradiance:     Optional[str] = Field(None, description="GHI column name in the raw CSV")
    temperature:    Optional[str] = None
    humidity:       Optional[str] = None
    wind_speed:     Optional[str] = None
    cloud_cover:    Optional[str] = None
    sunshine_hours: Optional[str] = None
    timestamp:      Optional[str] = Field(None, description="Datetime column name")


class UploadResponse(BaseModel):
    # Returned right after upload; the frontend uses session_id for all follow-up calls.
    session_id:       str   = Field(..., description="UUID identifying this upload session")
    filename:         str
    rows:             int   = Field(..., description="Number of data rows (excluding header)")
    columns_found:    int   = Field(..., description="Total columns in the raw file")
    detected_columns: DetectedColumns
    detection_mode:   DetectionMode
    confidence:       float = Field(..., ge=0.0, le=1.0, description="Claude API confidence (0–1)")
    preview:          list[dict] = Field(default_factory=list, description="First 5 rows as dicts")
    warnings:         list[str]  = Field(default_factory=list, description="Non-fatal detection issues")


class ForecastRequest(BaseModel):
    # Body sent by the frontend to POST /api/forecast/run.
    session_id: str
    horizon:    ForecastHorizon = ForecastHorizon.H24
    train_size: int = Field(80, ge=50, le=95, description="Training data percentage (50–95)")


class ModelMetrics(BaseModel):
    # Evaluation metrics for one model on the held-out test set.
    rmse: float = Field(..., description="Root Mean Square Error (W/m²)")
    mae:  float = Field(..., description="Mean Absolute Error (W/m²)")
    r2:   float = Field(..., description="Coefficient of Determination")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)")


class ForecastPoint(BaseModel):
    # A single time-step in the forecast; actual is None for future predictions.
    timestamp: datetime
    predicted: float         = Field(..., description="Ensemble prediction (W/m²)")
    actual:    Optional[float] = Field(None, description="Ground-truth value if available")
    lower:     Optional[float] = Field(None, description="Lower confidence bound (Prophet)")
    upper:     Optional[float] = Field(None, description="Upper confidence bound (Prophet)")


class PerModelForecast(BaseModel):
    # Raw predictions from one model — used in the Model Comparison page.
    model_name:    str
    predictions:   list[float]
    metrics:       ModelMetrics
    train_metrics: Optional[ModelMetrics] = None
    weight:        float = Field(..., ge=0.0, le=1.0, description="Ensemble weight for this model")


class ForecastResponse(BaseModel):
    # Full response from POST /api/forecast/run.
    session_id:         str
    run_id:             int  = Field(..., description="DB primary key (used in /history)")
    horizon:            int  = Field(..., description="Forecast horizon in hours")
    detection_mode:     DetectionMode
    forecast:           list[ForecastPoint]
    ensemble_metrics:   ModelMetrics
    per_model:          list[PerModelForecast]
    feature_importance: Optional[dict[str, float]] = None
    created_at:         datetime = Field(default_factory=datetime.utcnow)


class AnomalyRecord(BaseModel):
    # One detected anomaly in the irradiance series.
    timestamp: datetime
    value:     float  = Field(..., description="Observed irradiance (W/m²)")
    expected:  float  = Field(..., description="Expected / rolling-mean value")
    deviation: float  = Field(..., description="Absolute deviation from expected")
    severity:  AnomalySeverity
    method:    AnomalyMethod = Field(..., description="Detection method that flagged this point")


class AnomalyResponse(BaseModel):
    session_id:    str
    total_points:  int   = Field(..., description="Total data points scanned")
    anomaly_count: int
    anomaly_rate:  float = Field(..., description="anomaly_count / total_points")
    anomalies:     list[AnomalyRecord]


class ForecastRunSummary(BaseModel):
    # One row in /history — lightweight metadata only, no full forecast arrays.
    model_config = {"from_attributes": True}

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


class ErrorDetail(BaseModel):
    # Consistent error envelope for all endpoints — adds machine-readable code for the frontend.
    code:    str  = Field(..., description="Snake-case error code, e.g. 'column_not_found'")
    message: str
    field:   Optional[str] = Field(None, description="Which field caused the error, if any")


class HealthResponse(BaseModel):
    status:    str  = "ok"
    version:   str  = "1.0.0"
    db_online: bool

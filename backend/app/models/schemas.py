# Pydantic v2 schemas for all API request and response types.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# Enums

class DetectionMode(str, Enum):
    DIRECT = "direct"


class AnomalySeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AnomalyMethod(str, Enum):
    ZSCORE  = "zscore"
    IQR     = "iqr"
    ROLLING = "rolling"


class CorrectionSource(str, Enum):
    AI                     = "ai"
    PHYSICS_RULE           = "physics_rule"
    INTERPOLATION_FALLBACK = "interpolation_fallback"


class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# Upload 
class DetectedColumns(BaseModel):
    irradiance:     Optional[str] = Field(None, description="GHI column name in the raw CSV")
    temperature:    Optional[str] = None
    humidity:       Optional[str] = None
    wind_speed:     Optional[str] = None
    cloud_cover:    Optional[str] = None
    sunshine_hours: Optional[str] = None
    timestamp:      Optional[str] = Field(None, description="Datetime column name")


class ColumnQuality(BaseModel):
    name:          str
    missing:       int
    missing_pct:   float
    unique:        int
    min:           Optional[float] = None
    max:           Optional[float] = None
    mean:          Optional[float] = None
    std:           Optional[float] = None


class DataStats(BaseModel):
    rows_raw:          int
    rows_clean:        int
    rows_dropped:      int = 0
    pct_clean:         float
    columns_available: list[str]
    irradiance_mean:   float
    irradiance_max:    float
    irradiance_min:    float
    irradiance_std:    float = 0.0
    date_range_days:   int
    date_start:        str
    date_end:          str
    detection_mode:    Optional[str] = None
   
    duplicate_rows:      int = 0
    missing_hours:       int = 0
    total_missing_cells: int = 0
    clipped_count:       int = 0
    outlier_count:       int = 0
    column_quality:    list[ColumnQuality] = Field(default_factory=list)
  
    hourly_avg:        list[float] = Field(default_factory=list)
    weekday_avg:       list[float] = Field(default_factory=list)
    monthly_avg:       list[float] = Field(default_factory=list)
    daily_avg:         list[dict]  = Field(default_factory=list)


class UploadResponse(BaseModel):
   
    session_id:       str   = Field(..., description="UUID identifying this upload session")
    filename:         str
    rows:             int   = Field(..., description="Number of data rows (excluding header)")
    columns_found:    int   = Field(..., description="Total columns in the raw file")
    detected_columns: DetectedColumns
    detection_mode:   DetectionMode
    confidence:       float = Field(..., ge=0.0, le=1.0, description="Claude API confidence (0–1)")
    preview:          list[dict] = Field(default_factory=list, description="First 5 rows as dicts")
    warnings:         list[str]  = Field(default_factory=list, description="Non-fatal detection issues")
    data_stats:       Optional[DataStats] = None


# Forecast 
class ForecastRequest(BaseModel):
    
    session_id:  str
    horizon:     int  = Field(24, ge=1, le=168,  description="Forecast horizon in hours (1–168, max 1 week)")
    train_size:  int  = Field(80, ge=50, le=95,  description="Training data percentage (50–95)")
    skip_future: bool = Field(False, description="Skip future forecast generation (model evaluation only)")


class ModelMetrics(BaseModel):
   
    rmse: float = Field(..., description="Root Mean Square Error (W/m²)")
    mae:  float = Field(..., description="Mean Absolute Error (W/m²)")
    r2:   float = Field(..., description="Coefficient of Determination")


class ForecastPoint(BaseModel):
   
    timestamp: datetime
    predicted: float           = Field(..., description="Predicted irradiance (W/m²)")
    actual:    Optional[float] = Field(None, description="Ground-truth value if available")
    lower:     Optional[float] = Field(None, description="Lower confidence bound")
    upper:     Optional[float] = Field(None, description="Upper confidence bound")


class PerModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    #
    model_name:    str
    predictions:   list[float]
    metrics:       ModelMetrics
    train_metrics: Optional[ModelMetrics] = None
    is_best:       bool = False


class ForecastResponse(BaseModel):
    
    session_id:         str
    run_id:             int  = Field(..., description="DB primary key (used in /history)")
    horizon:            int  = Field(..., description="Forecast horizon in hours")
    detection_mode:     DetectionMode
    best_model:         str  = Field(..., description="Name of the best model used for forecasting")
    forecast:           list[ForecastPoint]
    future_forecast:    list[ForecastPoint] = Field(default_factory=list, description="True future predictions beyond last known data point")
    metrics:            ModelMetrics
    rows_processed:     int = Field(0, description="Number of training data points used")
    models_info:        list[PerModelInfo] = Field(default_factory=list)
    feature_importance: Optional[dict[str, float]] = None
    created_at:         datetime = Field(default_factory=datetime.utcnow)


# Anomaly Detection

class AnomalyRecord(BaseModel):
   
    timestamp: datetime
    value:     float = Field(..., description="Observed irradiance (W/m²)")
    expected:  float = Field(..., description="Expected / rolling-mean value")
    deviation: float = Field(..., description="Absolute deviation from expected")
    severity:  AnomalySeverity
    method:    AnomalyMethod = Field(..., description="Detection method that flagged this point")


class AnomalyResponse(BaseModel):
    session_id:    str
    total_points:  int   = Field(..., description="Total data points scanned")
    anomaly_count: int
    anomaly_rate:  float = Field(..., description="anomaly_count / total_points")
    anomalies:     list[AnomalyRecord]


# AI Correction 

class CorrectionEntry(BaseModel):
    timestamp:          str
    original_value:     float = Field(..., description="Raw sensor value that was flagged (W/m²)")
    corrected_value:    float = Field(..., description="AI- or rule-corrected replacement (W/m²)")
    reasoning:          str   = Field(..., description="Claude's 1-2 sentence physical explanation")
    confidence:         ConfidenceLevel
    method_flagged_by:  str   = Field(..., description="Which anomaly detector flagged this point")
    severity:           AnomalySeverity
    correction_source:  CorrectionSource = Field(
        ..., description="Who produced the correction: AI, physics rule, or interpolation fallback"
    )


class CorrectionStats(BaseModel):
    total_corrected:            int
    ai_corrections:             int
    physics_rule_corrections:   int
    interpolation_fallbacks:    int
    avg_correction_delta:       float = Field(..., description="Mean |original − corrected| (W/m²)")
    max_correction_delta:       float = Field(..., description="Largest single correction (W/m²)")
    high_confidence:            int
    medium_confidence:          int
    low_confidence:             int


class MetricsDelta(BaseModel):
    original:        float
    corrected:       float
    delta:           float = Field(..., description="corrected − original")
    improvement_pct: float = Field(..., description="% improvement (positive = better)")


class CorrectionRunResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
   
    session_id:          str
    correction_id:       str  = Field(..., description="UUID for this correction session (used in export)")
    anomaly_count:       int  = Field(..., description="Total anomalies detected before correction")
    anomalies_corrected: int  = Field(..., description="Number of points actually corrected")
    stats:               CorrectionStats
    correction_log:      list[CorrectionEntry]
   
    metrics_comparison:  dict[str, MetricsDelta] = Field(
        default_factory=dict,
        description="Metric deltas before/after correction — empty if pipeline was unavailable"
    )
   
    model_comparison:    dict[str, dict[str, ModelMetrics]] = Field(default_factory=dict)
    # forecasts: { "original": [...], "corrected": [...], "timestamps": [...] }
    forecasts:           dict[str, list] = Field(default_factory=dict)


class CorrectionLogResponse(BaseModel):
    
    correction_id: str
    session_id:    str
    total:         int
    corrections:   list[CorrectionEntry]


# History
class ForecastRunSummary(BaseModel):
   
    model_config = {"from_attributes": True}

    run_id:         int
    session_id:     str
    filename:       str
    horizon:        int
    detection_mode: DetectionMode
    best_model:     Optional[str] = None
    rmse:           float
    mae:            float
    r2:             float
    rows_processed: int
    anomaly_count:  int
    created_at:     datetime


class HistoryResponse(BaseModel):
    total_runs: int
    runs:       list[ForecastRunSummary]


class ErrorDetail(BaseModel):
    code:    str           = Field(..., description="Snake-case error code, e.g. 'column_not_found'")
    message: str
    field:   Optional[str] = Field(None, description="Which field caused the error, if any")


class HealthResponse(BaseModel):
    status:    str  = "ok"
    version:   str  = "1.0.0"
    db_online: bool
"""
pipeline.py — Wires all three ML models + ensemble into a single callable.

Flow:
  1. Receive clean DataFrame from preprocessing.py
  2. Run feature engineering (XGBoost + LightGBM)
  3. Train/predict all three models in parallel
  4. Compute ensemble weights from validation RMSE
  5. Return structured results matching ForecastResponse schema

Import path note:
  ml_core lives at the project root (one level above backend/).
  We add the project root to sys.path at import time so this works
  whether called from uvicorn, pytest, or a CLI script.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# ── Resolve project root so ml_core is importable ────────────────────────────
# backend/app/services/pipeline.py → go up 3 levels → project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── ML core imports ───────────────────────────────────────────────────────────
from ml_core.evaluate import compute_all, compare_models
from ml_core.feature_engineering import build_features, get_feature_columns
from ml_core.models.ensemble import EnsembleModel
from ml_core.models.lightgbm_model import LightGBMModel
from ml_core.models.prophet_model import ProphetModel
from ml_core.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_SIZE   = 0.20   # default — overridden per-request via train_size
TARGET_COL  = "irradiance"
PROPHET_DS  = "ds"
PROPHET_Y   = "y"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Train / test split (time-aware — no shuffling)
# ─────────────────────────────────────────────────────────────────────────────

def _time_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets preserving time order.
    Never shuffle time series data — future leakage destroys model validity.
    """
    split_idx = int(len(df) * (1 - test_size))
    train     = df.iloc[:split_idx]
    test      = df.iloc[split_idx:]
    logger.info(
        "Time split: train=%d rows, test=%d rows (%.0f%% / %.0f%%)",
        len(train), len(test),
        (1 - test_size) * 100, test_size * 100,
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature matrix builder (XGBoost + LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(
    df:      pd.DataFrame,
    col_map: dict,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a clean DataFrame.
    Returns a DataFrame with all engineered features + target column.
    """
    # preprocessing.py already sets DatetimeIndex — build_features expects it
    # Pass col_map for weather interaction features (cloud_clearness, etc.)
    feature_df = build_features(
        df,
        target_col    = TARGET_COL,
        timestamp_col = "timestamp",   # ignored if already index
        col_map       = col_map,
    )
    logger.info(
        "Feature matrix built: %d rows × %d features",
        len(feature_df),
        len(get_feature_columns(feature_df)),
    )
    return feature_df



# ─────────────────────────────────────────────────────────────────────────────
# 4. Individual model runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_xgboost(
    train_feat: pd.DataFrame,
    test_feat:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    feat_cols = get_feature_columns(train_feat)

    X_train = train_feat[feat_cols]
    y_train = train_feat[TARGET_COL].values
    X_test  = test_feat[feat_cols]
    y_test  = test_feat[TARGET_COL].values

    model = XGBoostModel()
    model.train(X_train, y_train)

    train_preds   = model.predict(X_train)
    test_preds    = model.predict(X_test)
    train_metrics = compute_all(y_train, train_preds)
    test_metrics  = compute_all(y_test,  test_preds)
    importance    = model.get_feature_importance() if hasattr(model, "get_feature_importance") else {}

    logger.info("XGBoost  | TRAIN RMSE=%.2f R²=%.3f | TEST RMSE=%.2f R²=%.3f",
                train_metrics["rmse"], train_metrics["r2"],
                test_metrics["rmse"],  test_metrics["r2"])
    return train_preds, test_preds, train_metrics, test_metrics, importance


def _run_lightgbm(
    train_feat: pd.DataFrame,
    test_feat:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    feat_cols = get_feature_columns(train_feat)

    X_train = train_feat[feat_cols]
    y_train = train_feat[TARGET_COL].values
    X_test  = test_feat[feat_cols]
    y_test  = test_feat[TARGET_COL].values

    model = LightGBMModel()
    model.train(X_train, y_train)

    train_preds   = model.predict(X_train)
    test_preds    = model.predict(X_test)
    train_metrics = compute_all(y_train, train_preds)
    test_metrics  = compute_all(y_test,  test_preds)
    importance    = model.get_feature_importance() if hasattr(model, "get_feature_importance") else {}

    logger.info("LightGBM | TRAIN RMSE=%.2f R²=%.3f | TEST RMSE=%.2f R²=%.3f",
                train_metrics["rmse"], train_metrics["r2"],
                test_metrics["rmse"],  test_metrics["r2"])
    return train_preds, test_preds, train_metrics, test_metrics, importance


def _run_prophet(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    horizon:  int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].values

    model = ProphetModel()
    model.train(X_train, y_train)

    # Predict on training set for train metrics
    train_preds = model.predict(X_train)
    train_preds = np.clip(train_preds, 0, None)
    train_metrics = compute_all(y_train.values, train_preds)

    # Predict on test set with intervals
    ci_df = model.predict_with_intervals(X_test)
    test_preds = np.clip(ci_df["yhat"].values,       0, None)
    lower      = np.clip(ci_df["yhat_lower"].values, 0, None)
    upper      = np.clip(ci_df["yhat_upper"].values, 0, None)

    test_metrics = compute_all(y_test, test_preds)

    logger.info("Prophet  | TRAIN RMSE=%.2f R²=%.3f | TEST RMSE=%.2f R²=%.3f",
                train_metrics["rmse"], train_metrics["r2"],
                test_metrics["rmse"],  test_metrics["r2"])
    return train_preds, test_preds, lower, upper, train_metrics, test_metrics

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ensemble weighting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ensemble_weights(metrics: dict[str, dict]) -> dict[str, float]:
    """
    Compute ensemble weights inversely proportional to validation RMSE.

    Lower RMSE → higher weight. Formula:
        weight_i = (1 / RMSE_i) / sum(1 / RMSE_j for all j)

    This is data-driven — the best-performing model on this specific
    dataset automatically gets the highest influence in the ensemble.
    """
    inverse_rmse = {
        name: 1.0 / max(m["rmse"], 1e-6)
        for name, m in metrics.items()
    }
    total = sum(inverse_rmse.values())
    weights = {name: round(v / total, 4) for name, v in inverse_rmse.items()}

    logger.info(
        "Ensemble weights | XGBoost=%.3f  LightGBM=%.3f  Prophet=%.3f",
        weights.get("xgboost",  0),
        weights.get("lightgbm", 0),
        weights.get("prophet",  0),
    )
    return weights


def _blend_predictions(
    predictions: dict[str, np.ndarray],
    weights:     dict[str, float],
) -> np.ndarray:
    """
    Compute weighted average of model predictions.
    All prediction arrays must have the same length.
    """
    blended = np.zeros(len(next(iter(predictions.values()))))
    for name, preds in predictions.items():
        blended += weights[name] * preds
    return np.clip(blended, 0, None)   # irradiance can't be negative


# ─────────────────────────────────────────────────────────────────────────────
# 6. Forecast point builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_forecast_points(
    test_df:    pd.DataFrame,
    ensemble:   np.ndarray,
    lower:      np.ndarray,
    upper:      np.ndarray,
) -> list[dict]:
    """
    Build the list of ForecastPoint dicts for the API response.
    Matches the ForecastPoint schema in schemas.py.
    """
    actuals = test_df[TARGET_COL].values
    points  = []

    for i, ts in enumerate(test_df.index):
        points.append({
            "timestamp": ts.isoformat(),
            "predicted": round(float(ensemble[i]), 4),
            "actual":    round(float(actuals[i]),  4),
            "lower":     round(float(lower[i]),    4),
            "upper":     round(float(upper[i]),    4),
        })

    return points


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature importance merger (XGBoost + LightGBM average)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_feature_importance(
    xgb_imp:  dict,
    lgbm_imp: dict,
) -> dict[str, float]:
    """
    Average feature importances from XGBoost and LightGBM.
    Normalise so all values sum to 1.0 for clean display on the frontend.
    """
    all_features = set(xgb_imp) | set(lgbm_imp)
    merged = {}

    for feat in all_features:
        xgb_val  = xgb_imp.get(feat,  0.0)
        lgbm_val = lgbm_imp.get(feat, 0.0)
        merged[feat] = (xgb_val + lgbm_val) / 2.0

    # Normalise
    total = sum(merged.values())
    if total > 0:
        merged = {k: round(v / total, 4) for k, v in merged.items()}

    # Sort by importance descending
    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Public interface — called by forecast router
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    df:             pd.DataFrame,
    col_map:        dict,
    horizon:        int = 24,
    detection_mode: str = "direct",
    train_size:     int = 80,
) -> dict[str, Any]:
    """
    Full ML pipeline: feature engineering → train → predict → ensemble.

    Args:
        df:             Clean DataFrame from preprocessing.preprocess()
                        Must have DatetimeIndex and 'irradiance' column.
        col_map:        Column mapping from ai_detector (for weather features)
                        e.g. {"cloud_cover": "cloud_cover", "temperature": "temperature"}
        horizon:        Forecast horizon in hours (24 / 48 / 72)
        detection_mode: "direct" | "estimated" — passed through to response

    Returns:
        {
            "forecast":           list[ForecastPoint dicts],
            "ensemble_metrics":   {rmse, mae, r2, mape},
            "per_model":          list of per-model result dicts,
            "feature_importance": {feature_name: importance_score},
            "detection_mode":     str,
            "horizon":            int,
            "rows_processed":     int,
        }
    """
    logger.info(
        "Pipeline started | rows=%d  horizon=%dh  mode=%s",
        len(df), horizon, detection_mode,
    )

    # ── Step 1: Train / test split ────────────────────────────────────────────
    train_df, test_df = _time_split(df, test_size=1.0 - train_size / 100.0)

    # ── Step 2: Feature engineering (tree models only) ────────────────────────
    # Note: build_features expects the index to already be a DatetimeIndex
    # preprocessing.py sets this, so we pass the df directly
    full_feat  = _build_feature_matrix(df,       col_map)
    train_feat = _build_feature_matrix(train_df, col_map)
    test_feat  = _build_feature_matrix(test_df,  col_map)

    # ── Step 3: Run all three models ──────────────────────────────────────────
    logger.info("Training XGBoost...")
    xgb_train_preds, xgb_preds, xgb_train_metrics, xgb_metrics, xgb_imp = _run_xgboost(train_feat, test_feat)

    logger.info("Training LightGBM...")
    lgbm_train_preds, lgbm_preds, lgbm_train_metrics, lgbm_metrics, lgbm_imp = _run_lightgbm(train_feat, test_feat)

    logger.info("Training Prophet...")
    prophet_train_preds, prophet_preds, lower, upper, prophet_train_metrics, prophet_metrics = _run_prophet(
        train_df, test_df, horizon
    )

    # ── Step 4: Ensemble ──────────────────────────────────────────────────────
    all_metrics = {
        "xgboost":  xgb_metrics,
        "lightgbm": lgbm_metrics,
        "prophet":  prophet_metrics,
    }
    weights = _compute_ensemble_weights(all_metrics)

    # Align all predictions to the same length (Prophet may differ slightly)
    min_len = min(len(xgb_preds), len(lgbm_preds), len(prophet_preds))
    predictions = {
        "xgboost":  xgb_preds[:min_len],
        "lightgbm": lgbm_preds[:min_len],
        "prophet":  prophet_preds[:min_len],
    }
    lower  = lower[:min_len]
    upper  = upper[:min_len]
    test_df_aligned = test_df.iloc[:min_len]

    ensemble_preds   = _blend_predictions(predictions, weights)
    ensemble_metrics = compute_all(
        test_df_aligned[TARGET_COL].values,
        ensemble_preds,
    )
    logger.info(
        "Ensemble | RMSE=%.2f  R²=%.3f",
        ensemble_metrics["rmse"], ensemble_metrics["r2"],
    )

    # ── Step 5: Build output ──────────────────────────────────────────────────
    forecast_points    = _build_forecast_points(
        test_df_aligned, ensemble_preds, lower, upper
    )
    feature_importance = _merge_feature_importance(xgb_imp, lgbm_imp)

    per_model = [
        {
            "model_name":    "XGBoost",
            "predictions":   xgb_preds[:min_len].tolist(),
            "metrics":       xgb_metrics,
            "train_metrics": xgb_train_metrics,
            "weight":        weights["xgboost"],
        },
        {
            "model_name":    "LightGBM",
            "predictions":   lgbm_preds[:min_len].tolist(),
            "metrics":       lgbm_metrics,
            "train_metrics": lgbm_train_metrics,
            "weight":        weights["lightgbm"],
        },
        {
            "model_name":    "Prophet",
            "predictions":   prophet_preds[:min_len].tolist(),
            "metrics":       prophet_metrics,
            "train_metrics": prophet_train_metrics,
            "weight":        weights["prophet"],
        },
    ]

    logger.info("Pipeline complete | forecast_points=%d", len(forecast_points))

    return {
        "forecast":           forecast_points,
        "ensemble_metrics":   ensemble_metrics,
        "per_model":          per_model,
        "feature_importance": feature_importance,
        "detection_mode":     detection_mode,
        "horizon":            horizon,
        "rows_processed":     len(df),
    }
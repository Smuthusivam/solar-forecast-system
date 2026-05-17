# ML pipeline — XGBoost + LightGBM. Trains on historical data,
# evaluates on a held-out test set, then optionally forecasts future hours.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml_core.evaluate import compute_all
from ml_core.feature_engineering import build_features, get_feature_columns
from ml_core.models.lightgbm_model import LightGBMModel
from ml_core.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

TARGET_COL = "irradiance"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_split(df: pd.DataFrame, test_size: float = 0.20):
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    logger.info("Time split: train=%d rows, test=%d rows", len(train), len(test))
    return train, test


def _build_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    feat = build_features(df, target_col=TARGET_COL, timestamp_col="timestamp", col_map=col_map)
    logger.info("Features: %d rows × %d cols", len(feat), len(get_feature_columns(feat)))
    return feat


def _train_model(Model, train_feat: pd.DataFrame, name: str):
    feat_cols = get_feature_columns(train_feat)
    X = train_feat[feat_cols]
    y = train_feat[TARGET_COL].values
    model = Model()
    model.train(X, y)
    train_preds = model.predict(X)
    train_metrics = compute_all(y, train_preds)
    importance = model.get_feature_importance() if hasattr(model, "get_feature_importance") else {}
    logger.info("%s | TRAIN RMSE=%.2f R²=%.3f", name, train_metrics["rmse"], train_metrics["r2"])
    return model, train_metrics, importance, feat_cols


def _test_model(model, test_feat: pd.DataFrame, feat_cols: list, name: str):
    # Use only columns that exist in test_feat; fill any missing lag columns with 0
    missing = [c for c in feat_cols if c not in test_feat.columns]
    if missing:
        logger.warning("%s: test set missing columns %s — filling with 0", name, missing)
        for c in missing:
            test_feat = test_feat.copy()
            test_feat[c] = 0.0
    X = test_feat[feat_cols]
    y = test_feat[TARGET_COL].values
    preds = np.clip(model.predict(X), 0, None)
    metrics = compute_all(y, preds)
    logger.info("%s | TEST  RMSE=%.2f R²=%.3f", name, metrics["rmse"], metrics["r2"])
    return preds, metrics


def _merge_importance(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    merged = {k: (a.get(k, 0) + b.get(k, 0)) / 2 for k in keys}
    total = sum(merged.values()) or 1
    return dict(sorted({k: round(v / total, 4) for k, v in merged.items()}.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# Future forecast generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_future_forecast(
    df: pd.DataFrame,
    col_map: dict,
    horizon: int,
    best_model,
    feat_cols: list,
    residual_std: float = 0.0,
) -> list[dict]:
    """
    Iteratively forecast `horizon` future hours beyond the last row in df
    using the single best model (lowest test RMSE).
    Confidence intervals widen with each step to reflect compounding uncertainty.
    """
    _LOOKBACK = 200
    working   = df.iloc[-_LOOKBACK:].copy()
    last_ts   = df.index[-1]
    future_points = []
    z95 = 1.96

    for step in range(1, horizon + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)

        new_row = working.iloc[[-1]].copy()
        new_row.index = [next_ts]
        new_row[TARGET_COL] = 0.0
        working = pd.concat([working, new_row])

        feat_df = _build_features(working, col_map)
        for c in feat_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0

        X_future  = feat_df[feat_cols].iloc[[-1]]
        predicted = float(np.clip(best_model.predict(X_future), 0, None)[0])

        if next_ts.hour < 5 or next_ts.hour >= 21:
            predicted = 0.0

        working.at[next_ts, TARGET_COL] = predicted

        # Uncertainty grows with forecast horizon — scale std by sqrt(step)
        interval = z95 * residual_std * float(np.sqrt(step))
        lower    = round(max(0.0, predicted - interval), 4)
        upper    = round(predicted + interval, 4)

        future_points.append({
            "timestamp": next_ts.isoformat(),
            "predicted": round(predicted, 4),
            "actual":    None,
            "lower":     lower if residual_std > 0 else None,
            "upper":     upper if residual_std > 0 else None,
        })

    logger.info("Future forecast: %d hourly points from %s", horizon, last_ts + pd.Timedelta(hours=1))
    return future_points


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    df:             pd.DataFrame,
    col_map:        dict,
    horizon:        int = 24,
    detection_mode: str = "direct",
    train_size:     int = 80,
    skip_future:    bool = False,
) -> dict[str, Any]:
    """
    Train XGBoost + LightGBM, evaluate on held-out test set, pick the best
    model by RMSE. Future forecast is only generated when skip_future=False.
    """
    logger.info("Pipeline started | rows=%d  horizon=%dh  mode=%s", len(df), horizon, detection_mode)

    train_df, test_df = _time_split(df, test_size=1.0 - train_size / 100.0)
    train_feat = _build_features(train_df, col_map)
    test_feat  = _build_features(test_df,  col_map)

    logger.info("Training XGBoost...")
    xgb_model,  xgb_train_metrics,  xgb_imp,  feat_cols = _train_model(XGBoostModel,  train_feat, "XGBoost")

    logger.info("Training LightGBM...")
    lgbm_model, lgbm_train_metrics, lgbm_imp, _         = _train_model(LightGBMModel, train_feat, "LightGBM")

    xgb_preds,  xgb_metrics  = _test_model(xgb_model,  test_feat, feat_cols, "XGBoost")
    lgbm_preds, lgbm_metrics = _test_model(lgbm_model, test_feat, feat_cols, "LightGBM")

    if xgb_metrics["rmse"] <= lgbm_metrics["rmse"]:
        best_model, best_preds, best_metrics, best_train_metrics, best_name = (
            xgb_model, xgb_preds, xgb_metrics, xgb_train_metrics, "XGBoost"
        )
    else:
        best_model, best_preds, best_metrics, best_train_metrics, best_name = (
            lgbm_model, lgbm_preds, lgbm_metrics, lgbm_train_metrics, "LightGBM"
        )
    logger.info("Best model: %s | RMSE=%.2f  R²=%.3f", best_name, best_metrics["rmse"], best_metrics["r2"])

    actuals = test_df[TARGET_COL].values

    # Compute residual std from test set — used for confidence intervals
    residuals   = actuals - best_preds
    residual_std = float(np.std(residuals))
    z95          = 1.96  # 95% confidence interval

    forecast_points = [
        {
            "timestamp": ts.isoformat(),
            "predicted": round(float(best_preds[i]), 4),
            "actual":    round(float(actuals[i]), 4),
            "lower":     round(max(0.0, float(best_preds[i]) - z95 * residual_std), 4),
            "upper":     round(float(best_preds[i]) + z95 * residual_std, 4),
        }
        for i, ts in enumerate(test_df.index)
    ]

    future_points = [] if skip_future else _build_future_forecast(
        df, col_map, horizon, best_model, feat_cols, residual_std
    )

    feature_importance = _merge_importance(xgb_imp, lgbm_imp)

    models_info = [
        {
            "model_name":    "XGBoost",
            "predictions":   xgb_preds.tolist(),
            "metrics":       xgb_metrics,
            "train_metrics": xgb_train_metrics,
            "is_best":       best_name == "XGBoost",
        },
        {
            "model_name":    "LightGBM",
            "predictions":   lgbm_preds.tolist(),
            "metrics":       lgbm_metrics,
            "train_metrics": lgbm_train_metrics,
            "is_best":       best_name == "LightGBM",
        },
    ]

    logger.info("Pipeline complete | test_points=%d  future_points=%d", len(forecast_points), len(future_points))

    return {
        "forecast":           forecast_points,
        "future_forecast":    future_points,
        "metrics":            best_metrics,
        "best_model":         best_name,
        "models_info":        models_info,
        "feature_importance": feature_importance,
        "detection_mode":     detection_mode,
        "horizon":            horizon,
        "rows_processed":     len(df),
    }

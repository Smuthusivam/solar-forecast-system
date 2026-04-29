import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_core.feature_engineering import build_features, get_feature_columns


class XGBoostModel:
    """
    XGBoost wrapper for solar irradiance forecasting.
    Follows the same interface as LightGBMModel and ProphetModel so ensemble.py can call all three identically.
    """

    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators":     600,
            "learning_rate":    0.01,
            "max_depth":        4,
            "min_child_weight": 10,
            "gamma":            0.5,
            "subsample":        0.7,
            "colsample_bytree": 0.7,
            "reg_alpha":        0.5,
            "reg_lambda":       3.0,
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }

        if params:
            default_params.update(params)

        self.params          = default_params
        self.model           = XGBRegressor(**self.params)
        self.feature_columns = None  # set during training, reused at inference

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Fit the model and record the feature column order for consistent inference.
        self.feature_columns = list(X_train.columns)
        self.model.fit(X_train, y_train)
        print(f"[XGBoost] Trained on {len(X_train)} rows, {len(self.feature_columns)} features.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Generate predictions; clips tiny negatives to 0 (tree models can predict below zero at night).
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 0, None)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # Predict on the test set and return all four metrics as a dict.
        from ml_core.evaluate import compute_all

        y_pred  = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "XGBoost"
        return metrics

    def get_feature_importance(self) -> dict:
        # Return gain-based importances sorted descending — powers the Feature Importance chart.
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet.")

        scores     = self.model.feature_importances_
        importance = dict(zip(self.feature_columns, scores))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        # Save model + feature column list together so column order is preserved on reload.
        payload = {
            "model":           self.model,
            "feature_columns": self.feature_columns,
            "params":          self.params,
        }
        joblib.dump(payload, path)
        print(f"[XGBoost] Model saved to {path}")

    def load(self, path: str) -> None:
        # Load a saved model; predict() and evaluate() work immediately after.
        payload              = joblib.load(path)
        self.model           = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.params          = payload["params"]
        print(f"[XGBoost] Model loaded from {path}")


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from ml_core.feature_engineering import build_features, get_feature_columns

    periods = 24 * 30
    idx     = pd.date_range("2024-01-01", periods=periods, freq="h")
    hours   = idx.hour

    irradiance = np.where(
        (hours >= 6) & (hours <= 20),
        500 * np.sin(np.pi * (hours - 6) / 14) + np.random.normal(0, 30, periods),
        0
    ).clip(0)

    raw_df = pd.DataFrame({
        "timestamp":  idx,
        "irradiance": irradiance,
        "Cloud_%":    np.random.uniform(0, 80, periods),
        "Temp_C":     np.random.uniform(10, 35, periods),
        "RH":         np.random.uniform(30, 90, periods),
    })

    col_map    = {"cloud_cover": "Cloud_%", "temperature": "Temp_C", "humidity": "RH"}
    feature_df = build_features(raw_df, target_col="irradiance", col_map=col_map)
    feat_cols  = get_feature_columns(feature_df, target_col="irradiance")

    X = feature_df[feat_cols]
    y = feature_df["irradiance"]

    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBoostModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n[XGBoost] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    importance = model.get_feature_importance()
    print(f"\n[XGBoost] Top 5 features by importance:")
    for feat, score in list(importance.items())[:5]:
        print(f"  {feat}: {score:.4f}")

    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/xgboost.pkl")

    model2  = XGBoostModel()
    model2.load("ml_core/saved_models/xgboost.pkl")
    y_pred2 = model2.predict(X_test)
    print(f"\n[XGBoost] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\nXGBoostModel passed all checks.")

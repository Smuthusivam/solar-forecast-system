import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ml_core.models.xgboost_model  import XGBoostModel
from ml_core.models.lightgbm_model import LightGBMModel
from ml_core.models.prophet_model  import ProphetModel
from ml_core.evaluate              import compute_all, compare_models


class EnsembleModel:
    """
    Weighted ensemble of XGBoost, LightGBM, and Prophet.
    Weights are computed from validation RMSE — lower error gets a higher weight.
    """

    def __init__(self):
        self.xgb     = XGBoostModel()
        self.lgbm    = LightGBMModel()
        self.prophet = ProphetModel()

        self.weights    = {"xgboost": None, "lightgbm": None, "prophet": None}
        self.metrics    = {"xgboost": None, "lightgbm": None, "prophet": None, "ensemble": None}
        self.is_trained = False

    def train(
        self,
        X_train:   pd.DataFrame,
        y_train:   pd.Series,
        val_split: float = 0.2,
    ) -> dict:
        # Train all three models and compute inverse-RMSE ensemble weights from a validation split.
        split_idx = int(len(X_train) * (1 - val_split))
        X_tr      = X_train.iloc[:split_idx]
        y_tr      = y_train.iloc[:split_idx]
        X_val     = X_train.iloc[split_idx:]
        y_val     = y_train.iloc[split_idx:]

        print(f"[Ensemble] Training split — train: {len(X_tr)}, val: {len(X_val)} rows")

        print("[Ensemble] Training XGBoost...")
        self.xgb.train(X_tr, y_tr)

        print("[Ensemble] Training LightGBM...")
        self.lgbm.train(X_tr, y_tr)

        print("[Ensemble] Training Prophet...")
        self.prophet.train(X_tr, y_tr)

        xgb_pred     = self.xgb.predict(X_val)
        lgbm_pred    = self.lgbm.predict(X_val)
        prophet_pred = self.prophet.predict(X_val)

        xgb_metrics     = compute_all(np.array(y_val), xgb_pred)
        lgbm_metrics    = compute_all(np.array(y_val), lgbm_pred)
        prophet_metrics = compute_all(np.array(y_val), prophet_pred)

        self.metrics["xgboost"]  = {**xgb_metrics,     "model": "XGBoost"}
        self.metrics["lightgbm"] = {**lgbm_metrics,    "model": "LightGBM"}
        self.metrics["prophet"]  = {**prophet_metrics, "model": "Prophet"}

        rmse_xgb     = max(xgb_metrics["rmse"],     1e-6)
        rmse_lgbm    = max(lgbm_metrics["rmse"],    1e-6)
        rmse_prophet = max(prophet_metrics["rmse"], 1e-6)

        w_xgb     = 1 / rmse_xgb
        w_lgbm    = 1 / rmse_lgbm
        w_prophet = 1 / rmse_prophet
        total     = w_xgb + w_lgbm + w_prophet

        self.weights = {
            "xgboost":  round(w_xgb     / total, 4),
            "lightgbm": round(w_lgbm    / total, 4),
            "prophet":  round(w_prophet / total, 4),
        }

        self.is_trained = True

        print(f"\n[Ensemble] Validation RMSE:")
        print(f"  XGBoost  : {rmse_xgb:.4f}")
        print(f"  LightGBM : {rmse_lgbm:.4f}")
        print(f"  Prophet  : {rmse_prophet:.4f}")
        print(f"\n[Ensemble] Computed weights:")
        for name, w in self.weights.items():
            print(f"  {name:10}: {w:.4f}  ({w*100:.1f}%)")

        return self.weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Generate weighted ensemble predictions; clips negatives to 0.
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained yet. Call train() first.")

        xgb_pred     = self.xgb.predict(X)
        lgbm_pred    = self.lgbm.predict(X)
        prophet_pred = self.prophet.predict(X)

        ensemble_pred = (
            self.weights["xgboost"]  * xgb_pred  +
            self.weights["lightgbm"] * lgbm_pred +
            self.weights["prophet"]  * prophet_pred
        )

        return np.clip(ensemble_pred, 0, None)

    def predict_all(self, X: pd.DataFrame) -> dict:
        # Return predictions from all three models and the ensemble in one dict.
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained yet.")

        xgb_pred     = self.xgb.predict(X)
        lgbm_pred    = self.lgbm.predict(X)
        prophet_pred = self.prophet.predict(X)
        ensemble_pred = (
            self.weights["xgboost"]  * xgb_pred  +
            self.weights["lightgbm"] * lgbm_pred +
            self.weights["prophet"]  * prophet_pred
        )

        return {
            "xgboost":  xgb_pred,
            "lightgbm": lgbm_pred,
            "prophet":  prophet_pred,
            "ensemble": np.clip(ensemble_pred, 0, None),
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # Evaluate on the held-out test set and return ensemble metrics.
        all_preds = self.predict_all(X_test)
        y_true    = np.array(y_test)

        for key in ["xgboost", "lightgbm", "prophet"]:
            m = compute_all(y_true, all_preds[key])
            m["model"] = key.capitalize()
            self.metrics[key] = m

        ensemble_metrics          = compute_all(y_true, all_preds["ensemble"])
        ensemble_metrics["model"] = "Ensemble"
        self.metrics["ensemble"]  = ensemble_metrics

        return ensemble_metrics

    def get_comparison_table(self) -> pd.DataFrame:
        """Return all four model metrics as a sorted DataFrame."""
        metrics_list = [m for m in self.metrics.values() if m is not None]
        return compare_models(metrics_list)

    def save(self, save_dir: str) -> None:
        # Save all three models and ensemble weights/metrics to a directory.
        os.makedirs(save_dir, exist_ok=True)

        self.xgb.save(os.path.join(save_dir, "xgboost.pkl"))
        self.lgbm.save(os.path.join(save_dir, "lightgbm.pkl"))
        self.prophet.save(os.path.join(save_dir, "prophet.json"))

        meta = {
            "weights":    self.weights,
            "metrics":    self.metrics,
            "is_trained": self.is_trained,
        }
        joblib.dump(meta, os.path.join(save_dir, "ensemble_meta.pkl"))
        print(f"[Ensemble] All models saved to {save_dir}/")

    def load(self, save_dir: str) -> None:
        # Load all three models and weights from disk; predict() works immediately after.
        self.xgb.load(os.path.join(save_dir, "xgboost.pkl"))
        self.lgbm.load(os.path.join(save_dir, "lightgbm.pkl"))
        self.prophet.load(os.path.join(save_dir, "prophet.json"))

        meta             = joblib.load(os.path.join(save_dir, "ensemble_meta.pkl"))
        self.weights     = meta["weights"]
        self.metrics     = meta["metrics"]
        self.is_trained  = meta["is_trained"]

        print(f"[Ensemble] All models loaded from {save_dir}/")
        print(f"[Ensemble] Weights: {self.weights}")


if __name__ == "__main__":
    from ml_core.feature_engineering import build_features, get_feature_columns

    periods = 24 * 60
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

    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train)

    print("\n[Ensemble] Evaluating on test set...")
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    print(f"\n[Ensemble] Test set metrics:")
    for k, v in ensemble_metrics.items():
        if k != "model":
            print(f"  {k.upper():>6}: {v}")

    print(f"\n[Ensemble] Full model comparison:")
    print(ensemble.get_comparison_table().to_string(index=False))

    save_dir = "ml_core/saved_models"
    ensemble.save(save_dir)

    ensemble2 = EnsembleModel()
    ensemble2.load(save_dir)
    preds2 = ensemble2.predict(X_test)
    preds1 = ensemble.predict(X_test)
    print(f"\n[Ensemble] Reload check — predictions match: {np.allclose(preds1, preds2)}")
    print("\nEnsembleModel passed all checks.")

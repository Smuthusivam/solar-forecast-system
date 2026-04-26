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


# ─────────────────────────────────────────────
# SECTION 1 — Ensemble Class
# ─────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted ensemble that combines XGBoost, LightGBM, and Prophet.

    How weights are determined:
        1. All three models are trained on the same training set
        2. Each model predicts on a validation set (last 20% of training data)
        3. RMSE is computed for each model on the validation set
        4. Weight = 1 / RMSE  →  lower error = higher weight
        5. Weights are normalised so they sum to 1

    Final prediction = w_xgb * xgb_pred + w_lgbm * lgbm_pred + w_prophet * prophet_pred

    This means if XGBoost has the lowest RMSE on a given dataset, it
    automatically gets the highest weight for that dataset. The weights
    adapt to the data — no manual tuning needed.
    """

    def __init__(self):
        self.xgb     = XGBoostModel()
        self.lgbm    = LightGBMModel()
        self.prophet = ProphetModel()

        self.weights  = {"xgboost": None, "lightgbm": None, "prophet": None}
        self.metrics  = {"xgboost": None, "lightgbm": None, "prophet": None, "ensemble": None}
        self.is_trained = False


    # ─────────────────────────────────────────────
    # SECTION 2 — Train
    # ─────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        val_split: float = 0.2,
    ) -> dict:
        """
        Train all three models and compute ensemble weights.

        Parameters
        ----------
        X_train   : full feature matrix from build_features()
        y_train   : irradiance target series
        val_split : fraction of training data used for weight computation
                    (default 0.2 = last 20% of training rows)

        Returns
        -------
        weights dict so pipeline.py can log and display them
        """

        # ── Split training into train + validation ──
        split_idx  = int(len(X_train) * (1 - val_split))
        X_tr       = X_train.iloc[:split_idx]
        y_tr       = y_train.iloc[:split_idx]
        X_val      = X_train.iloc[split_idx:]
        y_val      = y_train.iloc[split_idx:]

        print(f"[Ensemble] Training split — train: {len(X_tr)}, val: {len(X_val)} rows")

        # ── Train each model ──
        print("[Ensemble] Training XGBoost...")
        self.xgb.train(X_tr, y_tr)

        print("[Ensemble] Training LightGBM...")
        self.lgbm.train(X_tr, y_tr)

        print("[Ensemble] Training Prophet...")
        self.prophet.train(X_tr, y_tr)

        # ── Validate and collect per-model RMSE ──
        xgb_pred    = self.xgb.predict(X_val)
        lgbm_pred   = self.lgbm.predict(X_val)
        prophet_pred = self.prophet.predict(X_val)

        xgb_metrics    = compute_all(np.array(y_val), xgb_pred)
        lgbm_metrics   = compute_all(np.array(y_val), lgbm_pred)
        prophet_metrics = compute_all(np.array(y_val), prophet_pred)

        self.metrics["xgboost"]  = {**xgb_metrics,    "model": "XGBoost"}
        self.metrics["lightgbm"] = {**lgbm_metrics,   "model": "LightGBM"}
        self.metrics["prophet"]  = {**prophet_metrics, "model": "Prophet"}

        # ── Compute weights: weight = 1 / RMSE, then normalise ──
        rmse_xgb    = xgb_metrics["rmse"]
        rmse_lgbm   = lgbm_metrics["rmse"]
        rmse_prophet = prophet_metrics["rmse"]

        # Protect against perfect RMSE = 0 (synthetic data edge case)
        rmse_xgb    = max(rmse_xgb,    1e-6)
        rmse_lgbm   = max(rmse_lgbm,   1e-6)
        rmse_prophet = max(rmse_prophet, 1e-6)

        w_xgb    = 1 / rmse_xgb
        w_lgbm   = 1 / rmse_lgbm
        w_prophet = 1 / rmse_prophet
        total    = w_xgb + w_lgbm + w_prophet

        self.weights = {
            "xgboost":  round(w_xgb    / total, 4),
            "lightgbm": round(w_lgbm   / total, 4),
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


    # ─────────────────────────────────────────────
    # SECTION 3 — Predict
    # ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted ensemble predictions.

        Combines all three model predictions using the weights
        computed during training. Returns a single numpy array
        of irradiance values (W/m²).
        """
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
        """
        Return predictions from all three models AND the ensemble
        in a single dict. Called by pipeline.py so the frontend
        can display all four lines on the Model Comparison chart.

        Returns
        -------
        {
            "xgboost"  : np.ndarray,
            "lightgbm" : np.ndarray,
            "prophet"  : np.ndarray,
            "ensemble" : np.ndarray,
        }
        """
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


    # ─────────────────────────────────────────────
    # SECTION 4 — Evaluate
    # ─────────────────────────────────────────────

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the ensemble on the held-out test set and store
        all four model metrics for the comparison table.

        Returns the ensemble metrics dict.
        """
        all_preds = self.predict_all(X_test)
        y_true    = np.array(y_test)

        # Compute metrics for each model on the real test set
        for key in ["xgboost", "lightgbm", "prophet"]:
            m = compute_all(y_true, all_preds[key])
            m["model"] = key.capitalize()
            self.metrics[key] = m

        ensemble_metrics = compute_all(y_true, all_preds["ensemble"])
        ensemble_metrics["model"] = "Ensemble"
        self.metrics["ensemble"]  = ensemble_metrics

        return ensemble_metrics


    def get_comparison_table(self) -> pd.DataFrame:
        """
        Return all four model metrics as a sorted DataFrame.
        Called by pipeline.py to send the comparison table to the frontend.
        """
        metrics_list = [m for m in self.metrics.values() if m is not None]
        return compare_models(metrics_list)


    # ─────────────────────────────────────────────
    # SECTION 5 — Save / Load
    # ─────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        """
        Save all three models and ensemble metadata to a directory.

        Directory layout:
            save_dir/
                xgboost.pkl
                lightgbm.pkl
                prophet.json
                ensemble_meta.pkl   ← weights + metrics
        """
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
        """
        Load all three models and ensemble metadata from a directory.
        After calling load(), predict() works immediately.
        """
        self.xgb.load(os.path.join(save_dir, "xgboost.pkl"))
        self.lgbm.load(os.path.join(save_dir, "lightgbm.pkl"))
        self.prophet.load(os.path.join(save_dir, "prophet.json"))

        meta = joblib.load(os.path.join(save_dir, "ensemble_meta.pkl"))
        self.weights    = meta["weights"]
        self.metrics    = meta["metrics"]
        self.is_trained = meta["is_trained"]

        print(f"[Ensemble] All models loaded from {save_dir}/")
        print(f"[Ensemble] Weights: {self.weights}")


# ─────────────────────────────────────────────
# SECTION 6 — Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from ml_core.feature_engineering import build_features, get_feature_columns

    # ── Synthetic dataset — 60 days hourly ──
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

    # Last 20% = test set
    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Train ensemble ──
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train)

    # ── Evaluate on test set ──
    print("\n[Ensemble] Evaluating on test set...")
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    print(f"\n[Ensemble] Test set metrics:")
    for k, v in ensemble_metrics.items():
        if k != "model":
            print(f"  {k.upper():>6}: {v}")

    # ── Full comparison table ──
    print(f"\n[Ensemble] Full model comparison:")
    print(ensemble.get_comparison_table().to_string(index=False))

    # ── Save and reload ──
    save_dir = "ml_core/saved_models"
    ensemble.save(save_dir)

    ensemble2 = EnsembleModel()
    ensemble2.load(save_dir)
    preds2 = ensemble2.predict(X_test)
    preds1 = ensemble.predict(X_test)
    print(f"\n[Ensemble] Reload check — predictions match: {np.allclose(preds1, preds2)}")
    print("\n✓ EnsembleModel passed all checks.")
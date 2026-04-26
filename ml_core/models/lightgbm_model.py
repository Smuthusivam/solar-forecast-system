import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_core.feature_engineering import build_features, get_feature_columns


# ─────────────────────────────────────────────
# SECTION 1 — Model Class
# ─────────────────────────────────────────────

class LightGBMModel:
    """
    LightGBM wrapper for solar irradiance forecasting.

    LightGBM is 3-5x faster than XGBoost on the same data because it
    grows trees leaf-wise (best-first) rather than level-wise.
    It uses num_leaves instead of max_depth to control complexity.

    Same interface as XGBoostModel — ensemble.py calls both identically.
    """

    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators":     500,
            "learning_rate":    0.05,
            "num_leaves":       63,      # main complexity param (replaces max_depth)
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        -1,      # suppress all LightGBM logs
        }

        if params:
            default_params.update(params)

        self.params          = default_params
        self.model           = LGBMRegressor(**self.params)
        self.feature_columns = None


    # ─────────────────────────────────────────────
    # SECTION 2 — Train
    # ─────────────────────────────────────────────

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit LightGBM on the training feature matrix.
        Identical signature to XGBoostModel.train().
        """
        self.feature_columns = list(X_train.columns)
        self.model.fit(X_train, y_train)
        print(f"[LightGBM] Trained on {len(X_train)} rows, {len(self.feature_columns)} features.")


    # ─────────────────────────────────────────────
    # SECTION 3 — Predict
    # ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate irradiance predictions (W/m²).
        Clips negatives to 0 — same reasoning as XGBoost.
        """
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 0, None)


    # ─────────────────────────────────────────────
    # SECTION 4 — Evaluate
    # ─────────────────────────────────────────────

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Predict on test set and return all four metrics as a dict.
        """
        from ml_core.evaluate import compute_all

        y_pred  = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "LightGBM"
        return metrics


    # ─────────────────────────────────────────────
    # SECTION 5 — Feature Importance
    # ─────────────────────────────────────────────

    def get_feature_importance(self) -> dict:
        """
        Return feature importances sorted descending.
        LightGBM uses split count by default — we switch to 'gain'
        to stay consistent with XGBoost (gain = quality of splits,
        not just how often a feature was used).
        """
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet.")

        scores     = self.model.booster_.feature_importance(importance_type="gain")
        importance = dict(zip(self.feature_columns, scores))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


    # ─────────────────────────────────────────────
    # SECTION 6 — Save / Load
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "model":           self.model,
            "feature_columns": self.feature_columns,
            "params":          self.params,
        }
        joblib.dump(payload, path)
        print(f"[LightGBM] Model saved to {path}")

    def load(self, path: str) -> None:
        payload              = joblib.load(path)
        self.model           = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.params          = payload["params"]
        print(f"[LightGBM] Model loaded from {path}")


# ─────────────────────────────────────────────
# SECTION 7 — Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from ml_core.feature_engineering import build_features, get_feature_columns

    # ── Synthetic dataset — 30 days hourly ──
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

    # ── Train ──
    model = LightGBMModel()
    model.train(X_train, y_train)

    # ── Predict ──
    y_pred = model.predict(X_test)
    print(f"\n[LightGBM] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    # ── Feature importance ──
    importance = model.get_feature_importance()
    print(f"\n[LightGBM] Top 5 features by importance:")
    for feat, score in list(importance.items())[:5]:
        print(f"  {feat}: {score:.4f}")

    # ── Save and reload ──
    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/lightgbm.pkl")

    model2 = LightGBMModel()
    model2.load("ml_core/saved_models/lightgbm.pkl")
    y_pred2 = model2.predict(X_test)
    print(f"\n[LightGBM] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\n✓ LightGBMModel passed all checks.")
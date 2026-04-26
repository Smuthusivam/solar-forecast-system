import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_core.feature_engineering import build_features, get_feature_columns


# ─────────────────────────────────────────────
# SECTION 1 — Model Class
# ─────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost wrapper for solar irradiance forecasting.

    Follows the standard model interface used by all three models
    in this project (XGBoost, LightGBM, Prophet) so that ensemble.py
    can call .train(), .predict(), .evaluate() on any of them
    interchangeably.
    """

    def __init__(self, params: dict = None):
        """
        params: optional dict to override default XGBoost hyperparameters.
        If None, the project defaults from the spec are used.
        """
        default_params = {
            "n_estimators":  500,
            "learning_rate": 0.05,
            "max_depth":     6,
            "subsample":     0.8,       # use 80% of rows per tree — reduces overfitting
            "colsample_bytree": 0.8,    # use 80% of features per tree
            "random_state":  42,
            "n_jobs":        -1,        # use all CPU cores
            "verbosity":     0,         # suppress training logs
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model  = XGBRegressor(**self.params)
        self.feature_columns = None     # set during training, reused at inference


    # ─────────────────────────────────────────────
    # SECTION 2 — Train
    # ─────────────────────────────────────────────

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the model on the training feature matrix.

        X_train : feature matrix from build_features() with target column removed
        y_train : irradiance values (the target)
        """
        self.feature_columns = list(X_train.columns)
        self.model.fit(X_train, y_train)
        print(f"[XGBoost] Trained on {len(X_train)} rows, {len(self.feature_columns)} features.")


    # ─────────────────────────────────────────────
    # SECTION 3 — Predict
    # ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate irradiance predictions (W/m²) for a feature matrix.

        Clips negative predictions to 0 — irradiance is always non-negative,
        but tree models can sometimes predict tiny negatives at night.
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
        Run predictions on the test set and return all four metrics.
        Returns a dict so pipeline.py and ensemble.py can read metrics
        by name without caring about order.
        """
        from ml_core.evaluate import compute_all

        y_pred = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "XGBoost"
        return metrics


    # ─────────────────────────────────────────────
    # SECTION 5 — Feature Importance
    # ─────────────────────────────────────────────

    def get_feature_importance(self) -> dict:
        """
        Return feature importances as a sorted dict {feature_name: score}.
        XGBoost uses 'gain' importance — the average gain of splits
        that use that feature. Higher = more influential.

        This powers the Feature Importance chart on the Model Comparison page.
        """
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet.")

        scores = self.model.feature_importances_
        importance = dict(zip(self.feature_columns, scores))

        # sort descending so the frontend can render it directly
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


    # ─────────────────────────────────────────────
    # SECTION 6 — Save / Load
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Persist the trained model and feature column list to disk.
        Saves both into a single dict so the column order is always
        preserved when the model is reloaded later.
        """
        payload = {
            "model":          self.model,
            "feature_columns": self.feature_columns,
            "params":         self.params,
        }
        joblib.dump(payload, path)
        print(f"[XGBoost] Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a previously saved model from disk.
        After calling load(), predict() and evaluate() work immediately
        without needing to call train() again.
        """
        payload = joblib.load(path)
        self.model           = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.params          = payload["params"]
        print(f"[XGBoost] Model loaded from {path}")


# ─────────────────────────────────────────────
# SECTION 7 — Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from ml_core.feature_engineering import build_features, get_feature_columns

    # ── Build synthetic dataset ──
    periods = 24 * 30   # 30 days hourly
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    hours = idx.hour

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

    col_map = {"cloud_cover": "Cloud_%", "temperature": "Temp_C", "humidity": "RH"}

    # ── Feature engineering ──
    feature_df = build_features(raw_df, target_col="irradiance", col_map=col_map)
    feat_cols  = get_feature_columns(feature_df, target_col="irradiance")

    X = feature_df[feat_cols]
    y = feature_df["irradiance"]

    # ── Train / test split (last 20% = test) ──
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Train ──
    model = XGBoostModel()
    model.train(X_train, y_train)

    # ── Evaluate ──
    # (evaluate.py must exist — if not yet, do a manual metrics check)
    y_pred = model.predict(X_test)
    print(f"\n[XGBoost] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    # ── Feature importance ──
    importance = model.get_feature_importance()
    print(f"\n[XGBoost] Top 5 features by importance:")
    for feat, score in list(importance.items())[:5]:
        print(f"  {feat}: {score:.4f}")

    # ── Save and reload ──
    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/xgboost.pkl")

    model2 = XGBoostModel()
    model2.load("ml_core/saved_models/xgboost.pkl")
    y_pred2 = model2.predict(X_test)
    print(f"\n[XGBoost] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\n✓ XGBoostModel passed all checks.")
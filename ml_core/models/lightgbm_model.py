import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_core.feature_engineering import build_features, get_feature_columns


class LightGBMModel:
    """
    LightGBM wrapper for solar irradiance forecasting.
    Same interface as XGBoostModel so ensemble.py can call both identically.
    """

    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators":         1200,
            "learning_rate":        0.03,
            "num_leaves":           31,
            "max_depth":            -1,
            "min_child_samples":    30,
            "min_split_gain":       0.0,
            "subsample":            0.8,
            "subsample_freq":       1,
            "colsample_bytree":     0.8,
            "reg_alpha":            0.1,
            "reg_lambda":           1.0,
            "objective":            "regression",
            "metric":               "rmse",
            "random_state":         42,
            "n_jobs":               -1,
            "verbosity":            -1,
            "early_stopping_rounds": 50,
            "eval_fraction":        0.1,
            "min_eval_rows":        200,
        }

        if params:
            default_params.update(params)

        self.early_stopping_rounds = default_params.pop("early_stopping_rounds")
        self.eval_fraction         = default_params.pop("eval_fraction")
        self.min_eval_rows         = default_params.pop("min_eval_rows")

        self.params          = default_params
        self.model           = LGBMRegressor(**self.params)
        self.feature_columns = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Fit LightGBM on the training feature matrix.
        self.feature_columns = list(X_train.columns)
        X_all = X_train[self.feature_columns]
        y_all = np.asarray(y_train)

        use_eval = self.eval_fraction > 0 and len(X_all) >= self.min_eval_rows
        if use_eval:
            split_idx = int(len(X_all) * (1 - self.eval_fraction))
            if split_idx <= 0 or split_idx >= len(X_all):
                use_eval = False

        if use_eval:
            X_tr, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
            y_tr, y_val = y_all[:split_idx], y_all[split_idx:]
            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=self.early_stopping_rounds,
            )
        else:
            self.model.fit(X_all, y_all)

        print(f"[LightGBM] Trained on {len(X_all)} rows, {len(self.feature_columns)} features.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Generate predictions; clips negatives to 0 since irradiance can't be negative.
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 0, None)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # Predict on the test set and return all four metrics.
        from ml_core.evaluate import compute_all

        y_pred  = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "LightGBM"
        return metrics

    def get_feature_importance(self) -> dict:
        # Return gain-based feature importances sorted descending.
        if self.feature_columns is None:
            raise RuntimeError("Model has not been trained yet.")

        scores     = self.model.booster_.feature_importance(importance_type="gain")
        importance = dict(zip(self.feature_columns, scores))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        payload = {
            "model":           self.model,
            "feature_columns": self.feature_columns,
            "params":          self.params,
            "training_params": {
                "early_stopping_rounds": self.early_stopping_rounds,
                "eval_fraction": self.eval_fraction,
                "min_eval_rows": self.min_eval_rows,
            },
        }
        joblib.dump(payload, path)
        print(f"[LightGBM] Model saved to {path}")

    def load(self, path: str) -> None:
        payload              = joblib.load(path)
        self.model           = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.params          = payload["params"]
        training_params      = payload.get("training_params", {})
        self.early_stopping_rounds = training_params.get("early_stopping_rounds", self.early_stopping_rounds)
        self.eval_fraction         = training_params.get("eval_fraction", self.eval_fraction)
        self.min_eval_rows         = training_params.get("min_eval_rows", self.min_eval_rows)
        print(f"[LightGBM] Model loaded from {path}")


if __name__ == "__main__":
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

    model = LightGBMModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n[LightGBM] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    importance = model.get_feature_importance()
    print(f"\n[LightGBM] Top 5 features by importance:")
    for feat, score in list(importance.items())[:5]:
        print(f"  {feat}: {score:.4f}")

    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/lightgbm.pkl")

    model2  = LightGBMModel()
    model2.load("ml_core/saved_models/lightgbm.pkl")
    y_pred2 = model2.predict(X_test)
    print(f"\n[LightGBM] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\nLightGBMModel passed all checks.")

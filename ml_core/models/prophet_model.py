import numpy as np
import pandas as pd
import json
import os
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json


class ProphetModel:
    """
    Prophet wrapper for solar irradiance forecasting.
    Unlike XGBoost/LightGBM, Prophet only needs ds + y — it builds its own seasonality.
    It also naturally produces confidence intervals used for the dashboard's shaded bands.
    """

    def __init__(self, params: dict = None):
        default_params = {
            "daily_seasonality":       True,
            "yearly_seasonality":      True,
            "weekly_seasonality":      False,   # solar irradiance has no weekly pattern
            "seasonality_mode":        "multiplicative",  # amplitude scales seasonally
            "interval_width":          0.95,
            "changepoint_prior_scale": 0.01,    # low value stops trend from extrapolating wildly
            "seasonality_prior_scale": 10.0,
            "n_changepoints":          15,
            "changepoint_range":       0.8,     # only place changepoints in the first 80% of data
        }

        if params:
            default_params.update(params)

        self.params      = default_params
        self.model       = None   # created fresh in train()
        self.forecast_df = None   # stored after predict() so predict_with_intervals() can reuse it

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Fit Prophet using only the timestamp index and irradiance target.
        prophet_df = pd.DataFrame({
            "ds": X_train.index,
            "y":  y_train.values,
        })

        prophet_df["y"] = prophet_df["y"].clip(lower=0)

        self.model = Prophet(**self.params)

        # Suppress the verbose Stan output Prophet prints during fitting
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        self.model.fit(prophet_df)
        print(f"[Prophet] Trained on {len(prophet_df)} rows.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Generate point predictions for the given timestamps; clips negatives to 0.
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        future           = pd.DataFrame({"ds": X.index})
        self.forecast_df = self.model.predict(future)

        yhat = self.forecast_df["yhat"].values
        return np.clip(yhat, 0, None)

    def predict_with_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        # Return yhat + confidence interval columns for the dashboard chart bands.
        self.predict(X)

        result = self.forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["yhat"]       = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        return result

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # Predict on the test set and return metrics — same interface as XGBoost/LightGBM.
        from ml_core.evaluate import compute_all

        y_pred  = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "Prophet"
        return metrics

    def get_feature_importance(self) -> dict:
        # Return seasonality component strengths instead of feature importances.
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        components = {}
        for s in self.model.seasonalities:
            components[s] = round(self.model.seasonalities[s]["prior_scale"], 4)

        return components

    def save(self, path: str) -> None:
        # Prophet uses JSON serialisation, not pickle like the tree models.
        if self.model is None:
            raise RuntimeError("Nothing to save — model has not been trained yet.")

        with open(path, "w") as f:
            json.dump(model_to_json(self.model), f)

        print(f"[Prophet] Model saved to {path}")

    def load(self, path: str) -> None:
        # Load a saved Prophet model from its JSON file.
        with open(path, "r") as f:
            self.model = model_from_json(json.load(f))

        print(f"[Prophet] Model loaded from {path}")


if __name__ == "__main__":
    import sys
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
    })

    feature_df = build_features(raw_df, target_col="irradiance")

    X = feature_df.drop(columns=["irradiance"])
    y = feature_df["irradiance"]

    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = ProphetModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n[Prophet] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    ci_df = model.predict_with_intervals(X_test)
    print(f"\n[Prophet] Confidence interval sample (first 3 rows):")
    print(ci_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(3).to_string(index=False))

    components = model.get_feature_importance()
    print(f"\n[Prophet] Seasonality components: {components}")

    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/prophet.json")

    model2  = ProphetModel()
    model2.load("ml_core/saved_models/prophet.json")
    y_pred2 = model2.predict(X_test)
    print(f"\n[Prophet] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\nProphetModel passed all checks.")

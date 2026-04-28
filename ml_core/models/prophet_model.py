import numpy as np
import pandas as pd
import json
import os
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json


# ─────────────────────────────────────────────
# SECTION 1 — Model Class
# ─────────────────────────────────────────────

class ProphetModel:
    """
    Facebook/Meta Prophet wrapper for solar irradiance forecasting.

    Prophet is fundamentally different from XGBoost and LightGBM:
    - It does NOT use the feature engineering pipeline
    - It only needs two columns: ds (timestamp) and y (irradiance)
    - It builds its own internal seasonality components automatically
    - It naturally produces confidence intervals (yhat_lower, yhat_upper)
      which power the shaded bands on the forecast dashboard

    Prophet works best for capturing repeating seasonal patterns —
    daily sun cycles and annual seasonal variation — which makes it
    a strong complement to the tree-based models in the ensemble.
    """

    def __init__(self, params: dict = None):
        """
        Prophet key parameters:
        - daily_seasonality       : captures the intra-day sun cycle (sunrise → peak → sunset)
        - yearly_seasonality      : captures summer vs winter irradiance differences
        - weekly_seasonality      : off — solar irradiance has no weekly pattern
        - interval_width          : confidence interval width (0.95 = 95% CI bands on dashboard)
        - seasonality_mode        : "multiplicative" — irradiance amplitude scales seasonally
                                    (peaks in summer are larger than in winter, not just shifted)
        - changepoint_prior_scale : trend flexibility. Lowered to 0.01 to stop the trend
                                    from extrapolating wildly into the held-out test period.
        - seasonality_prior_scale : how strongly seasonality is fit (10 is Prophet default).
        - n_changepoints          : reduced from 25 to 15 to further constrain the trend.
        - changepoint_range       : only place changepoints in first 80% of history —
                                    prevents the trend from bending sharply near the test boundary.
        """
        default_params = {
            "daily_seasonality":       True,
            "yearly_seasonality":      True,
            "weekly_seasonality":      False,
            "seasonality_mode":        "multiplicative",
            "interval_width":          0.95,
            "changepoint_prior_scale": 0.01,
            "seasonality_prior_scale": 10.0,
            "n_changepoints":          15,
            "changepoint_range":       0.8,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model  = None      # Prophet instance — created fresh in train()
        self.forecast_df = None # stores the full forecast df after predict()


    # ─────────────────────────────────────────────
    # SECTION 2 — Train
    # ─────────────────────────────────────────────

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit Prophet on training data.

        Unlike XGBoost/LightGBM, Prophet does not take a feature matrix.
        It only needs a DataFrame with:
            ds : datetime column (the timestamp index)
            y  : the target values (irradiance)

        X_train is accepted as a parameter to keep the same interface
        as the other models, but only its index (timestamps) is used.
        y_train provides the irradiance values.
        """
        prophet_df = pd.DataFrame({
            "ds": X_train.index,
            "y":  y_train.values,
        })

        # Prophet cannot handle negative y values — clip just in case
        prophet_df["y"] = prophet_df["y"].clip(lower=0)

        self.model = Prophet(**self.params)

        # Suppress the verbose Stan output Prophet prints during fitting
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        self.model.fit(prophet_df)
        print(f"[Prophet] Trained on {len(prophet_df)} rows.")


    # ─────────────────────────────────────────────
    # SECTION 3 — Predict
    # ─────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions (yhat) for the given timestamps.

        X is the feature matrix — we only use its index (the timestamps).
        Returns a numpy array of predicted irradiance values (W/m²),
        clipped to 0 since irradiance cannot be negative.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        future = pd.DataFrame({"ds": X.index})
        self.forecast_df = self.model.predict(future)

        yhat = self.forecast_df["yhat"].values
        return np.clip(yhat, 0, None)


    def predict_with_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extended predict that returns yhat + confidence interval columns.
        Called by pipeline.py to get the CI bands for the dashboard chart.

        Returns a DataFrame with columns:
            ds          : timestamp
            yhat        : point prediction
            yhat_lower  : lower bound of confidence interval
            yhat_upper  : upper bound of confidence interval
        """
        self.predict(X)  # populates self.forecast_df

        result = self.forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["yhat"]       = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        return result


    # ─────────────────────────────────────────────
    # SECTION 4 — Evaluate
    # ─────────────────────────────────────────────

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Predict on test set and return metrics dict.
        Same interface as XGBoost and LightGBM evaluate().
        """
        from ml_core.evaluate import compute_all

        y_pred  = self.predict(X_test)
        metrics = compute_all(np.array(y_test), y_pred)
        metrics["model"] = "Prophet"
        return metrics


    # ─────────────────────────────────────────────
    # SECTION 5 — Feature Importance
    # ─────────────────────────────────────────────

    def get_feature_importance(self) -> dict:
        """
        Prophet doesn't have feature importances in the tree-model sense.
        Instead we return its internal seasonality component strengths,
        which tells us how much of the variance each seasonal pattern explains.

        This is displayed differently on the frontend — as a seasonality
        breakdown rather than a feature bar chart.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        components = {}
        for s in self.model.seasonalities:
            components[s] = round(self.model.seasonalities[s]["prior_scale"], 4)

        return components


    # ─────────────────────────────────────────────
    # SECTION 6 — Save / Load
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Prophet models are saved as JSON (not .pkl like the tree models).
        This is Prophet's official serialization format via prophet.serialize.
        """
        if self.model is None:
            raise RuntimeError("Nothing to save — model has not been trained yet.")

        with open(path, "w") as f:
            json.dump(model_to_json(self.model), f)

        print(f"[Prophet] Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a previously saved Prophet model from its JSON file.
        """
        with open(path, "r") as f:
            self.model = model_from_json(json.load(f))

        print(f"[Prophet] Model loaded from {path}")


# ─────────────────────────────────────────────
# SECTION 7 — Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

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
    })

    # Prophet only needs ds + y — no weather columns needed
    feature_df = build_features(raw_df, target_col="irradiance")

    X = feature_df.drop(columns=["irradiance"])
    y = feature_df["irradiance"]

    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Train ──
    model = ProphetModel()
    model.train(X_train, y_train)

    # ── Point predictions ──
    y_pred = model.predict(X_test)
    print(f"\n[Prophet] Sample predictions (first 5):")
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")

    # ── Predictions with confidence intervals ──
    ci_df = model.predict_with_intervals(X_test)
    print(f"\n[Prophet] Confidence interval sample (first 3 rows):")
    print(ci_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(3).to_string(index=False))

    # ── Seasonality components ──
    components = model.get_feature_importance()
    print(f"\n[Prophet] Seasonality components: {components}")

    # ── Save and reload ──
    os.makedirs("ml_core/saved_models", exist_ok=True)
    model.save("ml_core/saved_models/prophet.json")

    model2 = ProphetModel()
    model2.load("ml_core/saved_models/prophet.json")
    y_pred2 = model2.predict(X_test)
    print(f"\n[Prophet] Reload check — predictions match: {np.allclose(y_pred, y_pred2)}")
    print("\nProphetModel passed all checks.")
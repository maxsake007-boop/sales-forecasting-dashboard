# ============================================================
# src/model.py — Reusable model functions
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

FEATURES = [
    "dayofweek", "month", "quarter", "year",
    "is_weekend", "day",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7"
]
TARGET = "revenue"


def split_train_test(df: pd.DataFrame, test_days: int = 90):
    """
    Split dataset by time. Last N days = test, rest = train.
    """
    split_date = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] > split_date]
    return train, test


def evaluate_model(name: str, y_true, y_pred) -> dict:
    """
    Calculate MAE, MAPE, RMSE for a model.
    Returns a dict with results.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "model": name,
        "MAE":   round(mae, 2),
        "MAPE":  round(mape * 100, 2),
        "RMSE":  round(rmse, 2)
    }


def train_models(X_train, y_train) -> dict:
    """
    Train all three models: LinearRegression, XGBoost, LightGBM.
    Returns a dict with trained model objects.
    """
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                       subsample=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)

    lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                         subsample=0.8, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    return {
        "Linear Regression": baseline,
        "XGBoost":           xgb,
        "LightGBM":          lgbm
    }


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame sorted by MAPE.
    """
    results = []
    for name, model in models.items():
        pred = model.predict(X_test)
        results.append(evaluate_model(name, y_test, pred))
    return pd.DataFrame(results).set_index("model").sort_values("MAPE")


def get_best_model(models: dict, X_test, y_test):
    """
    Return the best model by MAPE.
    """
    results_df = compare_models(models, X_test, y_test)
    best_name  = results_df["MAPE"].idxmin()
    return best_name, models[best_name]


def save_model(model, path: str = "models/forecast_model_v1.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str = "models/forecast_model_v1.pkl"):
    """Load trained model from disk."""
    return joblib.load(path)


def forecast_30_days(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 30-day forecast with realistic, optimistic, and pessimistic scenarios.
    Returns a DataFrame with columns: date, forecast, optimistic, pessimistic.
    """
    last_date = df["date"].max()
    last_30   = list(df["revenue"].tail(30))

    forecast_rows = []
    for i in range(1, 31):
        future_date = last_date + pd.Timedelta(days=i)
        lag_7_val   = last_30[-7]  if len(last_30) >= 7  else np.mean(last_30)
        lag_14_val  = last_30[-14] if len(last_30) >= 14 else np.mean(last_30)
        lag_30_val  = last_30[-30] if len(last_30) >= 30 else np.mean(last_30)

        row = {
            "date":            future_date,
            "dayofweek":       future_date.dayofweek,
            "month":           future_date.month,
            "quarter":         future_date.quarter,
            "year":            future_date.year,
            "is_weekend":      int(future_date.dayofweek in [5, 6]),
            "day":             future_date.day,
            "lag_7":           lag_7_val,
            "lag_14":          lag_14_val,
            "lag_30":          lag_30_val,
            "rolling_mean_7":  np.mean(last_30[-7:]),
            "rolling_mean_14": np.mean(last_30[-14:]),
            "rolling_mean_30": np.mean(last_30[-30:]),
            "rolling_std_7":   np.std(last_30[-7:]),
        }
        forecast_rows.append(row)

        pred_val = model.predict(pd.DataFrame([row])[FEATURES])[0]
        last_30.append(pred_val)
        last_30 = last_30[-30:]

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["forecast"]    = model.predict(forecast_df[FEATURES])
    forecast_df["optimistic"]  = forecast_df["forecast"] * 1.15
    forecast_df["pessimistic"] = forecast_df["forecast"] * 0.85

    return forecast_df[["date", "forecast", "optimistic", "pessimistic"]]
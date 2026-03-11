# ============================================================
# scripts/run_model.py — Model Training & Evaluation
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── 1. LOAD FEATURE DATASET ───────────────────────────────────
df = pd.read_csv("data/processed/retail_store_sales_features.csv")
df["date"] = pd.to_datetime(df["date"])

print("=" * 55)
print("MODEL TRAINING — SALES FORECAST")
print("=" * 55)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

# ── 2. DEFINE FEATURES & TARGET ───────────────────────────────
FEATURES = [
    "dayofweek", "month", "quarter", "year",
    "is_weekend", "day",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7"
]
TARGET = "revenue"

# ── 3. TRAIN / TEST SPLIT (by time) ───────────────────────────
# Last 90 days = test, everything before = train
split_date = df["date"].max() - pd.Timedelta(days=90)

train = df[df["date"] <= split_date]
test  = df[df["date"] > split_date]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\n[INFO] Train: {len(train)} days ({train['date'].min().date()} → {train['date'].max().date()})")
print(f"[INFO] Test:  {len(test)} days  ({test['date'].min().date()} → {test['date'].max().date()})")

# ── 4. HELPER: evaluate model ─────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"\n  {name}")
    print(f"    MAE:  ${mae:,.2f}")
    print(f"    MAPE: {mape:.2%}")
    print(f"    RMSE: ${rmse:,.2f}")
    return {"model": name, "MAE": round(mae, 2), "MAPE": round(mape * 100, 2), "RMSE": round(rmse, 2)}

results = []

# ── 5. BASELINE — LINEAR REGRESSION ──────────────────────────
print("\n" + "=" * 55)
print("BASELINE — LINEAR REGRESSION")
print("=" * 55)

baseline = LinearRegression()
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)
results.append(evaluate("Linear Regression", y_test, pred_baseline))

# ── 6. XGBOOST ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("XGBOOST")
print("=" * 55)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                   subsample=0.8, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
results.append(evaluate("XGBoost", y_test, pred_xgb))

# ── 7. LIGHTGBM ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("LIGHTGBM")
print("=" * 55)

lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                     subsample=0.8, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
pred_lgbm = lgbm.predict(X_test)
results.append(evaluate("LightGBM", y_test, pred_lgbm))

# ── 8. MODELS COMPARISON ──────────────────────────────────────
print("\n" + "=" * 55)
print("MODELS COMPARISON")
print("=" * 55)
results_df = pd.DataFrame(results).set_index("model")
print(results_df.to_string())

best_model_name = results_df["MAPE"].idxmin()
print(f"\n[INFO] Best model by MAPE: {best_model_name}")

# ── 9. SAVE BEST MODEL ────────────────────────────────────────
best_model = {"Linear Regression": baseline, "XGBoost": xgb, "LightGBM": lgbm}[best_model_name]
joblib.dump(best_model, "models/forecast_model_v1.pkl")
print(f"[INFO] Best model saved → models/forecast_model_v1.pkl")

# ── 10. FORECAST NEXT 30 DAYS ─────────────────────────────────
print("\n" + "=" * 55)
print("FORECAST — NEXT 30 DAYS")
print("=" * 55)

last_date    = df["date"].max()
last_revenue = df["revenue"].values
last_30      = list(df["revenue"].tail(30))

forecast_rows = []
for i in range(1, 31):
    future_date = last_date + pd.Timedelta(days=i)
    lag_7_val   = last_30[-7]  if len(last_30) >= 7  else np.mean(last_30)
    lag_14_val  = last_30[-14] if len(last_30) >= 14 else np.mean(last_30)
    lag_30_val  = last_30[-30] if len(last_30) >= 30 else np.mean(last_30)

    roll_7  = np.mean(last_30[-7:])
    roll_14 = np.mean(last_30[-14:])
    roll_30 = np.mean(last_30[-30:])
    std_7   = np.std(last_30[-7:])

    row = {
        "date":             future_date,
        "dayofweek":        future_date.dayofweek,
        "month":            future_date.month,
        "quarter":          future_date.quarter,
        "year":             future_date.year,
        "is_weekend":       int(future_date.dayofweek in [5, 6]),
        "day":              future_date.day,
        "lag_7":            lag_7_val,
        "lag_14":           lag_14_val,
        "lag_30":           lag_30_val,
        "rolling_mean_7":   roll_7,
        "rolling_mean_14":  roll_14,
        "rolling_mean_30":  roll_30,
        "rolling_std_7":    std_7,
    }
    forecast_rows.append(row)

    # Add predicted value to rolling window for next iteration
    pred_val = best_model.predict(pd.DataFrame([row])[FEATURES])[0]
    last_30.append(pred_val)
    last_30 = last_30[-30:]

forecast_df = pd.DataFrame(forecast_rows)
forecast_df["forecast"]     = best_model.predict(forecast_df[FEATURES])
forecast_df["optimistic"]   = forecast_df["forecast"] * 1.15
forecast_df["pessimistic"]  = forecast_df["forecast"] * 0.85

total_forecast   = forecast_df["forecast"].sum()
total_optimistic = forecast_df["optimistic"].sum()
total_pessimistic= forecast_df["pessimistic"].sum()

print(f"Realistic forecast  (30 days): ${total_forecast:,.2f}")
print(f"Optimistic forecast (30 days): ${total_optimistic:,.2f}")
print(f"Pessimistic forecast(30 days): ${total_pessimistic:,.2f}")

# ── 11. SAVE FORECAST CSV ─────────────────────────────────────
export_df = forecast_df[["date", "forecast", "optimistic", "pessimistic"]].copy()
export_df.columns = ["Date", "Forecast ($)", "Optimistic ($)", "Pessimistic ($)"]
export_df = export_df.round(2)
export_df.to_csv("outputs/forecast_30days.csv", index=False)
print(f"\n[INFO] Forecast saved → outputs/forecast_30days.csv")

# ── 12. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# --- Plot 1: Actual vs Predicted (test period) ---
axes[0, 0].plot(test["date"], y_test.values, color="#BFDBFE", linewidth=1, label="Actual")
axes[0, 0].plot(test["date"], pred_xgb, color="#2563EB", linewidth=1.5, label="XGBoost")
axes[0, 0].plot(test["date"], pred_lgbm, color="#1E3A8A", linewidth=1.5, linestyle="--", label="LightGBM")
axes[0, 0].set_title("Actual vs Predicted — Test Period", fontsize=13)
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0, 0].legend(fontsize=8)

# --- Plot 2: 30-Day Forecast with scenarios ---
axes[0, 1].fill_between(forecast_df["date"],
                         forecast_df["pessimistic"],
                         forecast_df["optimistic"],
                         alpha=0.2, color="#2563EB", label="Pessimistic / Optimistic range")
axes[0, 1].plot(forecast_df["date"], forecast_df["forecast"], color="#2563EB", linewidth=2, label="Realistic")
axes[0, 1].set_title("30-Day Revenue Forecast", fontsize=13)
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Forecast Revenue ($)")
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0, 1].legend(fontsize=8)

# --- Plot 3: Feature Importance (XGBoost) ---
fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
axes[1, 0].barh(fi.index, fi.values, color="#2563EB")
axes[1, 0].set_title("Feature Importance — XGBoost", fontsize=13)
axes[1, 0].set_xlabel("Importance Score")

# --- Plot 4: Models Comparison (MAPE) ---
axes[1, 1].bar(results_df.index, results_df["MAPE"], color="#2563EB")
axes[1, 1].set_title("Models Comparison — MAPE (%)", fontsize=13)
axes[1, 1].set_ylabel("MAPE (%)")
for i, (idx, row) in enumerate(results_df.iterrows()):
    axes[1, 1].text(i, row["MAPE"] + 0.3, f"{row['MAPE']}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/model_results.png", dpi=150)
plt.show()
print("[INFO] Chart saved → outputs/model_results.png")

print("\n" + "=" * 55)
print("DONE. Model training complete.")
print("=" * 55)


# ── SHAP ──────────────────────────────────────────────────────
import shap

print("\n[INFO] Running SHAP analysis...")
explainer   = shap.Explainer(xgb)
shap_values = explainer(X_train)

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP — Feature Impact on Revenue Forecast")
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP — Mean Feature Importance")
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] SHAP charts saved → outputs/shap_summary.png, shap_bar.png")

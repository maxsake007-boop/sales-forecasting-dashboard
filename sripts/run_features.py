# ============================================================
# scripts/run_features.py — Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── LOAD CLEAN DATA ───────────────────────────────────────────
df = pd.read_csv("data/processed/retail_store_sales_clean.csv")
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])

print("=" * 55)
print("FEATURE ENGINEERING")
print("=" * 55)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1. AGGREGATE TO DAILY REVENUE ────────────────────────────
# Model needs one row per day — sum all transactions per day
daily = df.groupby("Transaction Date")["Total Spent"].sum().reset_index()
daily.columns = ["date", "revenue"]
daily = daily.sort_values("date").reset_index(drop=True)

print(f"\n[INFO] Aggregated to daily level: {len(daily)} days")
print(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")

# ── 2. CALENDAR FEATURES ─────────────────────────────────────
daily["dayofweek"]  = daily["date"].dt.dayofweek      # 0=Monday, 6=Sunday
daily["month"]      = daily["date"].dt.month
daily["quarter"]    = daily["date"].dt.quarter
daily["year"]       = daily["date"].dt.year
daily["is_weekend"] = daily["dayofweek"].isin([5, 6]).astype(int)
daily["day"]        = daily["date"].dt.day

print("\n[INFO] Calendar features created:")
print("       dayofweek, month, quarter, year, is_weekend, day")

# ── 3. LAG FEATURES ───────────────────────────────────────────
# Revenue from N days ago — model sees what happened in the past
daily["lag_7"]  = daily["revenue"].shift(7)
daily["lag_14"] = daily["revenue"].shift(14)
daily["lag_30"] = daily["revenue"].shift(30)

print("\n[INFO] Lag features created:")
print("       lag_7, lag_14, lag_30")

# ── 4. ROLLING MEAN FEATURES ──────────────────────────────────
# Smoothed average over last N days — captures trend
daily["rolling_mean_7"]  = daily["revenue"].shift(1).rolling(7).mean()
daily["rolling_mean_14"] = daily["revenue"].shift(1).rolling(14).mean()
daily["rolling_mean_30"] = daily["revenue"].shift(1).rolling(30).mean()

print("\n[INFO] Rolling mean features created:")
print("       rolling_mean_7, rolling_mean_14, rolling_mean_30")

# ── 5. ROLLING STD ────────────────────────────────────────────
# Volatility — how much revenue fluctuates
daily["rolling_std_7"] = daily["revenue"].shift(1).rolling(7).std()

print("\n[INFO] Rolling std feature created:")
print("       rolling_std_7")

# ── 6. DROP ROWS WITH NaN FROM LAGS ──────────────────────────
rows_before = len(daily)
daily = daily.dropna().reset_index(drop=True)
rows_after = len(daily)

print(f"\n[INFO] Dropped {rows_before - rows_after} rows with NaN from lags.")
print(f"       Rows remaining: {rows_after}")

# ── 7. FINAL OVERVIEW ─────────────────────────────────────────
print("\n" + "=" * 55)
print("FEATURE DATASET OVERVIEW")
print("=" * 55)
print(f"Shape: {daily.shape[0]} rows × {daily.shape[1]} columns")
print(f"\nColumns:")
for col in daily.columns:
    print(f"  {col}")

print(f"\nSample (last 5 rows):")
print(daily.tail().to_string())

print(f"\nMissing values:")
print(daily.isnull().sum()[daily.isnull().sum() > 0])
if daily.isnull().sum().sum() == 0:
    print("  None — dataset is clean.")

# ── 8. QUICK VISUAL CHECK ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Revenue + Rolling Mean
axes[0, 0].plot(daily["date"], daily["revenue"], color="#BFDBFE", linewidth=0.8, label="Daily Revenue")
axes[0, 0].plot(daily["date"], daily["rolling_mean_7"], color="#2563EB", linewidth=1.5, label="Rolling Mean 7d")
axes[0, 0].plot(daily["date"], daily["rolling_mean_30"], color="#1E3A8A", linewidth=1.5, label="Rolling Mean 30d", linestyle="--")
axes[0, 0].set_title("Revenue + Rolling Averages")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].legend(fontsize=8)

# Lag 7 vs Revenue
axes[0, 1].scatter(daily["lag_7"], daily["revenue"], alpha=0.3, color="#2563EB", s=5)
axes[0, 1].set_title("Lag 7 vs Revenue (correlation check)")
axes[0, 1].set_xlabel("Revenue 7 Days Ago ($)")
axes[0, 1].set_ylabel("Revenue ($)")

# Revenue by Day of Week
dow_avg = daily.groupby("dayofweek")["revenue"].mean()
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1, 0].bar([day_names[i] for i in dow_avg.index], dow_avg.values, color="#2563EB")
axes[1, 0].set_title("Avg Revenue by Day of Week")
axes[1, 0].set_ylabel("Avg Revenue ($)")

# Revenue by Month
month_avg = daily.groupby("month")["revenue"].mean()
axes[1, 1].bar(month_avg.index, month_avg.values, color="#2563EB")
axes[1, 1].set_title("Avg Revenue by Month")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Avg Revenue ($)")

plt.tight_layout()
plt.savefig("outputs/features_overview.png", dpi=150)
plt.show()
print("\n[INFO] Chart saved → outputs/features_overview.png")

# ── 9. SAVE FEATURE DATASET ───────────────────────────────────
daily.to_csv("data/processed/retail_store_sales_features.csv", index=False)
print(f"\n[SUCCESS] Feature dataset saved → data/processed/retail_store_sales_features.csv")
print(f"Final shape: {daily.shape[0]} rows × {daily.shape[1]} columns")

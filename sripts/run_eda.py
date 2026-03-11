# ============================================================
# scripts/run_eda.py — Exploratory Data Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── LOAD CLEAN DATA ───────────────────────────────────────────
df = pd.read_csv("data/processed/retail_store_sales_clean.csv")
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])

print("=" * 55)
print("EDA — RETAIL STORE SALES")
print("=" * 55)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['Transaction Date'].min().date()} → {df['Transaction Date'].max().date()}")
print(f"Total Revenue: ${df['Total Spent'].sum():,.2f}")
print(f"Unique Customers: {df['Customer ID'].nunique()}")
print(f"Unique Categories: {df['Category'].nunique()}")

# ── HELPER: save figure ───────────────────────────────────────
def save_fig(filename):
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=150)
    plt.show()
    print(f"[INFO] Saved → outputs/{filename}")

# ============================================================
# 1. DAILY REVENUE TREND
# ============================================================
print("\n" + "=" * 55)
print("1. DAILY REVENUE TREND")
print("=" * 55)

daily = df.groupby("Transaction Date")["Total Spent"].sum()

# Rolling average to smooth the trend
daily_smooth = daily.rolling(7).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily.index, daily.values, color="#BFDBFE", linewidth=0.8, label="Daily Revenue")
ax.plot(daily_smooth.index, daily_smooth.values, color="#2563EB", linewidth=2, label="7-Day Rolling Avg")
ax.set_title("Daily Revenue Trend", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Total Spent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
save_fig("01_daily_revenue_trend.png")

print(f"Average daily revenue:  ${daily.mean():,.2f}")
print(f"Max daily revenue:      ${daily.max():,.2f}  on {daily.idxmax().date()}")
print(f"Min daily revenue:      ${daily.min():,.2f}  on {daily.idxmin().date()}")

# ============================================================
# 2. MONTHLY REVENUE
# ============================================================
print("\n" + "=" * 55)
print("2. MONTHLY REVENUE")
print("=" * 55)

df["month_period"] = df["Transaction Date"].dt.to_period("M")
monthly = df.groupby("month_period")["Total Spent"].sum()

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(monthly.index.astype(str), monthly.values, color="#2563EB", width=0.7)
ax.set_title("Monthly Revenue", fontsize=14)
ax.set_xlabel("Month")
ax.set_ylabel("Total Spent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right")
save_fig("02_monthly_revenue.png")

print(f"Best month:  {monthly.idxmax()}  →  ${monthly.max():,.2f}")
print(f"Worst month: {monthly.idxmin()}  →  ${monthly.min():,.2f}")

# ============================================================
# 3. REVENUE BY DAY OF WEEK
# ============================================================
print("\n" + "=" * 55)
print("3. REVENUE BY DAY OF WEEK")
print("=" * 55)

day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["dayofweek"] = df["Transaction Date"].dt.dayofweek
dow = df.groupby("dayofweek")["Total Spent"].sum()
dow.index = [day_names[i] for i in dow.index]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(dow.index, dow.values, color="#2563EB")
ax.set_title("Revenue by Day of Week", fontsize=14)
ax.set_xlabel("Day")
ax.set_ylabel("Total Spent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save_fig("03_revenue_by_day_of_week.png")

print(dow.sort_values(ascending=False).to_string())

# ============================================================
# 4. REVENUE BY CATEGORY
# ============================================================
print("\n" + "=" * 55)
print("4. REVENUE BY CATEGORY")
print("=" * 55)

cat_revenue = df.groupby("Category")["Total Spent"].sum().sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(cat_revenue.index, cat_revenue.values, color="#2563EB")
ax.set_title("Total Revenue by Category", fontsize=14)
ax.set_xlabel("Total Revenue ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Add value labels
for bar, val in zip(bars, cat_revenue.values):
    ax.text(val + 500, bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}", va="center", fontsize=9)
save_fig("04_revenue_by_category.png")

print(cat_revenue.sort_values(ascending=False).to_string())

# ============================================================
# 5. TOP 10 ITEMS BY REVENUE
# ============================================================
print("\n" + "=" * 55)
print("5. TOP 10 ITEMS BY REVENUE")
print("=" * 55)

top_items = df.groupby("Item")["Total Spent"].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_items.index[::-1], top_items.values[::-1], color="#2563EB")
ax.set_title("Top 10 Items by Revenue", fontsize=14)
ax.set_xlabel("Total Revenue ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save_fig("05_top_10_items.png")

print(top_items.to_string())

# ============================================================
# 6. TOP CUSTOMERS BY REVENUE
# ============================================================
print("\n" + "=" * 55)
print("6. TOP CUSTOMERS BY REVENUE")
print("=" * 55)

cust_revenue = df.groupby("Customer ID").agg(
    Total_Revenue=("Total Spent", "sum"),
    Num_Transactions=("Transaction ID", "count"),
    Avg_Order=("Total Spent", "mean")
).sort_values("Total_Revenue", ascending=False)

print(cust_revenue.head(10).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(cust_revenue.index, cust_revenue["Total_Revenue"], color="#2563EB")
ax.set_title("Revenue by Customer", fontsize=14)
ax.set_xlabel("Customer ID")
ax.set_ylabel("Total Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right")
save_fig("06_revenue_by_customer.png")

# ============================================================
# 7. REVENUE BY LOCATION & PAYMENT METHOD
# ============================================================
print("\n" + "=" * 55)
print("7. REVENUE BY LOCATION & PAYMENT METHOD")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Location
loc = df.groupby("Location")["Total Spent"].sum().sort_values()
axes[0].barh(loc.index, loc.values, color="#2563EB")
axes[0].set_title("Revenue by Location", fontsize=13)
axes[0].set_xlabel("Total Revenue ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Payment Method
pay = df.groupby("Payment Method")["Total Spent"].sum().sort_values()
axes[1].barh(pay.index, pay.values, color="#2563EB")
axes[1].set_title("Revenue by Payment Method", fontsize=13)
axes[1].set_xlabel("Total Revenue ($)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

save_fig("07_location_and_payment.png")

print("Location breakdown:")
print(loc.sort_values(ascending=False).to_string())
print("\nPayment Method breakdown:")
print(pay.sort_values(ascending=False).to_string())

# ============================================================
# 8. DISCOUNT IMPACT
# ============================================================
print("\n" + "=" * 55)
print("8. DISCOUNT IMPACT ON REVENUE")
print("=" * 55)

discount_impact = df.groupby("Discount Applied")["Total Spent"].agg(["mean", "sum", "count"])
discount_impact.index = ["No Discount", "Discount Applied"]
print(discount_impact.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(discount_impact.index, discount_impact["mean"], color="#2563EB")
axes[0].set_title("Avg Order Value — Discount vs No Discount", fontsize=12)
axes[0].set_ylabel("Avg Total Spent ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

axes[1].bar(discount_impact.index, discount_impact["count"], color="#2563EB")
axes[1].set_title("Number of Transactions — Discount vs No Discount", fontsize=12)
axes[1].set_ylabel("Transaction Count")

save_fig("08_discount_impact.png")

# ============================================================
# 9. DISTRIBUTION OF ORDER VALUES
# ============================================================
print("\n" + "=" * 55)
print("9. ORDER VALUE DISTRIBUTION")
print("=" * 55)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["Total Spent"], bins=50, color="#2563EB", edgecolor="white")
ax.set_title("Distribution of Order Values", fontsize=14)
ax.set_xlabel("Total Spent ($)")
ax.set_ylabel("Number of Orders")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Add median line
median_val = df["Total Spent"].median()
ax.axvline(median_val, color="red", linestyle="--", linewidth=1.5, label=f"Median: ${median_val:,.0f}")
ax.legend()
save_fig("09_order_value_distribution.png")

print(f"Mean order value:   ${df['Total Spent'].mean():,.2f}")
print(f"Median order value: ${df['Total Spent'].median():,.2f}")
print(f"Std dev:            ${df['Total Spent'].std():,.2f}")


# ── HELPER: save figure ───────────────────────────────────────
def save_fig(filename):
    """
    Сохраняет график в директорию outputs/PNG/
    """
    import os
    # Автоматическое создание директории, если она отсутствует
    os.makedirs("outputs/PNG/", exist_ok=True)

    plt.tight_layout()
    # Изменен путь сохранения
    path = f"outputs/PNG/{filename}"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[INFO] Saved → {path}")




# ============================================================
# 10. EDA SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("EDA SUMMARY")
print("=" * 55)
print(f"Total Revenue:         ${df['Total Spent'].sum():,.2f}")
print(f"Total Transactions:    {len(df):,}")
print(f"Avg Order Value:       ${df['Total Spent'].mean():,.2f}")
print(f"Top Category:          {cat_revenue.idxmax()}")
print(f"Top Customer:          {cust_revenue['Total_Revenue'].idxmax()}")
print(f"Best Day of Week:      {dow.idxmax()}")
print(f"Best Month:            {monthly.idxmax()}")
print("\n[SUCCESS] EDA complete. All charts saved to outputs/")
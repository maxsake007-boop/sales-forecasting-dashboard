# ============================================================
# RETAIL STORE SALES — DATA CLEANING & VALIDATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("data/raw/retail_store_sales.csv")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# ── 2. DATA TYPES & BASIC INFO ────────────────────────────────
print("\n" + "=" * 55)
print("DATA TYPES & NON-NULL COUNTS")
print("=" * 55)
df.info()

# ── 3. MISSING VALUES ANALYSIS ───────────────────────────────
print("\n" + "=" * 55)
print("MISSING VALUES")
print("=" * 55)

missing = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing %": (df.isna().sum() / len(df) * 100).round(2)
})
missing = missing[missing["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(missing)

# ── 4. DUPLICATES ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("DUPLICATES")
print("=" * 55)
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
print(f"Duplicate Transaction IDs: {df['Transaction ID'].duplicated().sum()}")

# ── 5. BASIC STATISTICS ───────────────────────────────────────
print("\n" + "=" * 55)
print("DESCRIPTIVE STATISTICS")
print("=" * 55)
print(df.describe())



# ── 6. REPLACE 'None' STRINGS WITH NaN ───────────────────────
df.replace("None", np.nan, inplace=True)
df.replace("nan", np.nan, inplace=True)  # ← добавь эту строку
df["Item"] = df["Item"].replace("nan", np.nan)  # ← и эту на всякий случай
print("\n[INFO] Replaced string 'None' and 'nan' values with NaN.")

print("\n" + "=" * 55)
print(" 7-8-9-10-11-12")
print("=" * 55)

# ── 7. FIX DATA TYPES ─────────────────────────────────────────
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
df["Price Per Unit"]   = pd.to_numeric(df["Price Per Unit"], errors="coerce")
df["Quantity"]         = pd.to_numeric(df["Quantity"], errors="coerce")
df["Total Spent"]      = pd.to_numeric(df["Total Spent"], errors="coerce")


print("[INFO] Data types corrected.")
print(df.dtypes)
# Standardize category names to title case
df['Category'] = df['Category'].str.strip().str.title()
print(df['Category'].unique())

# ── 8. REMOVE DUPLICATES ──────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\n[INFO] Removed {before - after} duplicate rows. Rows remaining: {after:,}")

# ── 9. DEDUCE MISSING 'Item' ──────────────────────────────────


# Force replace any remaining string 'nan' in Item column
df["Item"] = df["Item"].where(df["Item"] != "nan", np.nan)
print(f"NaN in Item before deduction: {df['Item'].isna().sum()}")
# Price Per Unit is static per item — build a lookup table
price_to_item = df.dropna(subset=["Item", "Price Per Unit"]) \
                  .drop_duplicates(subset=["Price Per Unit", "Category"]) \
                  .set_index(["Price Per Unit", "Category"])["Item"].to_dict()

def deduce_item(row):
    if pd.notna(row["Item"]):
        return row["Item"]
    key = (row["Price Per Unit"], row["Category"])
    return price_to_item.get(key, np.nan)

df["Item"] = df.apply(deduce_item, axis=1)
print(f"[INFO] Deduced missing Item values. Remaining NaN in Item: {df['Item'].isna().sum()}")

# ── 10. DEDUCE MISSING QUANTITY ───────────────────────────────
# If Total Spent and Price Per Unit are known → Quantity = Total / Price
mask_qty = df["Quantity"].isna() & df["Total Spent"].notna() & df["Price Per Unit"].notna()
df.loc[mask_qty, "Quantity"] = (df.loc[mask_qty, "Total Spent"] / df.loc[mask_qty, "Price Per Unit"]).round(0)
print(f"[INFO] Deduced {mask_qty.sum()} missing Quantity values.")


# ── 11. DEDUCE MISSING Total Spent ────────────────────────────
mask_total = df["Total Spent"].isna() & df["Quantity"].notna() & df["Price Per Unit"].notna()
df.loc[mask_total, "Total Spent"] = df.loc[mask_total, "Quantity"] * df.loc[mask_total, "Price Per Unit"]
print(f"[INFO] Deduced {mask_total.sum()} missing Total Spent values.")

# ── 12. DEDUCE MISSING Price Per Unit ────────────────────────
mask_price = df["Price Per Unit"].isna() & df["Quantity"].notna() & df["Total Spent"].notna()
df.loc[mask_price, "Price Per Unit"] = (df.loc[mask_price, "Total Spent"] / df.loc[mask_price, "Quantity"]).round(2)
print(f"[INFO] Deduced {mask_price.sum()} missing Price Per Unit values.")


print("\n" + "=" * 55)
print(" 13-17")
print("=" * 55)

# ── 13. VALIDATE CONSISTENCY ──────────────────────────────────
# Total Spent should equal Quantity * Price Per Unit (tolerance ±0.01)
df["_expected_total"] = (df["Quantity"] * df["Price Per Unit"]).round(2)
inconsistent = df[
    df["_expected_total"].notna() &
    df["Total Spent"].notna() &
    (abs(df["Total Spent"] - df["_expected_total"]) > 0.01)
]

print(f"\n[INFO] Inconsistent Total Spent rows: {len(inconsistent)}")
if len(inconsistent) > 0:
    print(inconsistent[["Transaction ID", "Quantity", "Price Per Unit", "Total Spent", "_expected_total"]].head(10))
    # Fix inconsistencies — trust Quantity * Price Per Unit
    df.loc[inconsistent.index, "Total Spent"] = df.loc[inconsistent.index, "_expected_total"]
    print("[INFO] Inconsistent Total Spent values corrected.")

df.drop(columns=["_expected_total"], inplace=True)

# ── 14. CLIP NEGATIVE VALUES ──────────────────────────────────
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    neg = (df[col] < 0).sum()
    if neg > 0:
        print(f"[WARNING] {neg} negative values found in '{col}' — clipping to 0.")
        df[col] = df[col].clip(lower=0)

# ── 15. FIX PAYMENT METHOD & LOCATION ────────────────────────
df["Payment Method"] = df["Payment Method"].str.strip().str.title()
df["Location"]       = df["Location"].str.strip().str.title()

print("Payment Method:", df["Payment Method"].unique())
print("Location:", df["Location"].unique())

valid_payments  = ["Cash", "Credit Card", "Debit Card", "Online Transfer", "Digital Wallet"]
valid_locations = ["In-Store", "Online"]

df.loc[~df["Payment Method"].isin(valid_payments), "Payment Method"] = np.nan
df.loc[~df["Location"].isin(valid_locations), "Location"] = np.nan

print("\n[INFO] Invalid categorical values replaced with NaN.")


# ── 16. FIX DISCOUNT APPLIED ─────────────────────────────────
df["Discount Applied"] = df["Discount Applied"].map(
    {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
)
print(f"[INFO] 'Discount Applied' unique values after fix: {df['Discount Applied'].unique()}")

# ── 17. DROP ROWS WITH NO USEFUL INFO ────────────────────────
# Rows where Total Spent is still NaN are unusable for forecasting
rows_before = len(df)
df = df.dropna(subset=["Total Spent"])
print(f"\n[INFO] Dropped {rows_before - len(df)} rows with no Total Spent. Rows remaining: {len(df):,}")

print("\n" + "=" * 55)
print(" 18-21 ")
print("=" * 55)

# ── 18. SORT BY DATE ──────────────────────────────────────────
df = df.sort_values("Transaction Date").reset_index(drop=True)
print("[INFO] DataFrame sorted by Transaction Date.")

# ── 19. FINAL MISSING VALUES SUMMARY ─────────────────────────
print("\n" + "=" * 55)
print("FINAL MISSING VALUES AFTER CLEANING")
print("=" * 55)
final_missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
})

print(final_missing[final_missing["Missing Count"] > 0])

# ── 20. SAVE CLEAN DATASET ────────────────────────────────────
df.to_csv("data/processed/retail_store_sales_clean.csv", index=False)
print("\n[SUCCESS] Clean dataset saved as 'retail_store_sales_clean.csv'")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 21. QUICK VISUAL CHECK ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Daily revenue trend
daily = df.groupby("Transaction Date")["Total Spent"].sum()
axes[0].plot(daily.index, daily.values, color="#2563EB", linewidth=1)
axes[0].set_title("Daily Revenue — After Cleaning", fontsize=13)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Total Spent ($)")


# Revenue by category
cat_revenue = df.groupby("Category")["Total Spent"].sum().sort_values()
axes[1].barh(cat_revenue.index, cat_revenue.values, color="#2563EB")
axes[1].set_title("Revenue by Category", fontsize=13)
axes[1].set_xlabel("Total Revenue ($)")

plt.tight_layout()
plt.savefig("outputs/PNG/cleaning_summary.png", dpi=150)
plt.show()
print("[INFO] Summary chart saved as 'cleaning_summary.png'")


# Check what Payment Method and Location look like before replacing
print(df['Payment Method'].value_counts(dropna=False))
print(df['Location'].value_counts(dropna=False))
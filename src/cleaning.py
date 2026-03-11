# ============================================================
# src/cleaning.py — Reusable cleaning functions
# ============================================================

import pandas as pd
import numpy as np


def replace_string_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace string 'None' and 'nan' with real NaN values."""
    df = df.replace("None", np.nan)
    df = df.replace("nan", np.nan)
    df["Item"] = df["Item"].where(df["Item"] != "nan", np.nan)
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column data types and standardize category names."""
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df["Price Per Unit"]   = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df["Quantity"]         = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Total Spent"]      = pd.to_numeric(df["Total Spent"], errors="coerce")
    df["Category"]         = df["Category"].str.strip().str.title()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def deduce_missing_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce missing Item values using Price Per Unit + Category lookup.
    Since each item has a static price, we can reverse-lookup the item name.
    """
    price_to_item = (
        df.dropna(subset=["Item", "Price Per Unit"])
        .drop_duplicates(subset=["Price Per Unit", "Category"])
        .set_index(["Price Per Unit", "Category"])["Item"]
        .to_dict()
    )

    def _lookup(row):
        if pd.notna(row["Item"]):
            return row["Item"]
        return price_to_item.get((row["Price Per Unit"], row["Category"]), np.nan)

    df["Item"] = df.apply(_lookup, axis=1)
    return df


def deduce_missing_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce missing Quantity, Total Spent, and Price Per Unit
    using the relationship: Total Spent = Quantity * Price Per Unit.
    """
    # Deduce Quantity
    mask_qty = df["Quantity"].isna() & df["Total Spent"].notna() & df["Price Per Unit"].notna()
    df.loc[mask_qty, "Quantity"] = (
        df.loc[mask_qty, "Total Spent"] / df.loc[mask_qty, "Price Per Unit"]
    ).round(0)

    # Deduce Total Spent
    mask_total = df["Total Spent"].isna() & df["Quantity"].notna() & df["Price Per Unit"].notna()
    df.loc[mask_total, "Total Spent"] = (
        df.loc[mask_total, "Quantity"] * df.loc[mask_total, "Price Per Unit"]
    )

    # Deduce Price Per Unit
    mask_price = df["Price Per Unit"].isna() & df["Quantity"].notna() & df["Total Spent"].notna()
    df.loc[mask_price, "Price Per Unit"] = (
        df.loc[mask_price, "Total Spent"] / df.loc[mask_price, "Quantity"]
    ).round(2)

    return df


def validate_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that Total Spent == Quantity * Price Per Unit.
    Fix inconsistencies by recalculating Total Spent.
    """
    df["_expected_total"] = (df["Quantity"] * df["Price Per Unit"]).round(2)
    inconsistent = df[
        df["_expected_total"].notna() &
        df["Total Spent"].notna() &
        (abs(df["Total Spent"] - df["_expected_total"]) > 0.01)
    ]
    if len(inconsistent) > 0:
        df.loc[inconsistent.index, "Total Spent"] = df.loc[inconsistent.index, "_expected_total"]
    df.drop(columns=["_expected_total"], inplace=True)
    return df, len(inconsistent)


def clip_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative values in numeric columns to 0."""
    for col in ["Price Per Unit", "Quantity", "Total Spent"]:
        df[col] = df[col].clip(lower=0)
    return df


def fix_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Payment Method and Location values.
    Replace invalid entries with NaN.
    """
    df["Payment Method"] = df["Payment Method"].str.strip().str.title()
    df["Location"]       = df["Location"].str.strip().str.title()

    valid_payments  = ["Cash", "Credit Card", "Debit Card", "Online Transfer", "Digital Wallet"]
    valid_locations = ["In-Store", "Online"]

    df.loc[~df["Payment Method"].isin(valid_payments), "Payment Method"] = np.nan
    df.loc[~df["Location"].isin(valid_locations), "Location"]            = np.nan

    return df


def fix_discount_applied(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Discount Applied column to boolean."""
    df["Discount Applied"] = df["Discount Applied"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    )
    return df


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Total Spent is missing — unusable for forecasting."""
    return df.dropna(subset=["Total Spent"]).reset_index(drop=True)


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by Transaction Date ascending."""
    return df.sort_values("Transaction Date").reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning function — runs all steps in order.
    Use this in Streamlit or other scripts for a one-liner clean.
    """
    df = replace_string_nulls(df)
    df = fix_data_types(df)
    df = remove_duplicates(df)
    df = deduce_missing_items(df)
    df = deduce_missing_numerics(df)
    df, _ = validate_consistency(df)
    df = clip_negative_values(df)
    df = fix_categoricals(df)
    df = fix_discount_applied(df)
    df = drop_unusable_rows(df)
    df = sort_by_date(df)
    return df
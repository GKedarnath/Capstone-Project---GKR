"""Shared data cleaning and feature engineering for shipment delay models."""
from __future__ import annotations
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
    "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw exports, remove exact duplicates and repair known sentinels."""
    df = raw.copy()
    df = df.drop_duplicates().copy()
    if "service_type" in df:
        df["service_type"] = df["service_type"].astype("string").str.strip().str.title()
    if "payment_mode" in df:
        df["payment_mode"] = df["payment_mode"].fillna("Unknown").astype(str).str.strip().str.title()
    if "declared_value" in df:
        df["declared_value_inr"] = pd.to_numeric(
            df["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        df = df.drop(columns=["declared_value"])
    elif "declared_value_inr" not in df:
        df["declared_value_inr"] = np.nan
    if "dispatch_time" in df:
        df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    if "parcel_weight_kg" in df:
        df.loc[pd.to_numeric(df["parcel_weight_kg"], errors="coerce") > 100, "parcel_weight_kg"] = np.nan
    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr", "n_stops", "rain_flag", "sale_event_flag"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_impute_values(cleaned: pd.DataFrame) -> dict:
    """Calculate training-set medians; reuse these values at inference time."""
    values = {}
    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr", "n_stops"]:
        median = cleaned[col].median() if col in cleaned else np.nan
        values[col] = float(median) if pd.notna(median) else 0.0
    return values


def build_features(cleaned: pd.DataFrame, hubs: pd.DataFrame, impute_values: dict | None = None) -> pd.DataFrame:
    """Join hub metadata and create the fixed numeric feature set."""
    df = cleaned.copy()
    medians = impute_values or calculate_impute_values(df)
    for col, value in medians.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(value)
    hub_cols = ["hub_id", "hub_tier", "hub_daily_capacity", "hub_city", "region"]
    available = [c for c in hub_cols if c in hubs.columns]
    df = df.merge(hubs[available].drop_duplicates("hub_id"), on="hub_id", how="left", validate="many_to_one")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).fillna(PROMISED_HOURS["Standard"])
    df["log_declared_value"] = np.log1p(df["declared_value_inr"].clip(lower=0))
    df["is_cod"] = (df["payment_mode"].str.upper() == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.fillna(12).astype(int)
    for col in FEATURES:
        if col not in df:
            df[col] = 0
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

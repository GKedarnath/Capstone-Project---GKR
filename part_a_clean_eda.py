"""Part A: clean the generated export, engineer features and write EDA outputs."""
from pathlib import Path
import pandas as pd
from src.shipment_prep import clean_shipments, calculate_impute_values, build_features, FEATURES
ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"

def main():
    OUT.mkdir(exist_ok=True)
    hubs = pd.read_csv(DATA / "hubs.csv")
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    cleaned = clean_shipments(raw)
    impute = calculate_impute_values(cleaned)
    features = build_features(cleaned, hubs, impute)
    if "delivery_hours" in cleaned:
        features["delivery_hours"] = pd.to_numeric(cleaned["delivery_hours"], errors="coerce").values
        features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)
    features.to_csv(DATA / "modelling_table.csv", index=False)
    pd.DataFrame({"column": raw.columns, "missing_values": [raw[c].isna().sum() for c in raw.columns]}).to_csv(OUT / "missing_values.csv", index=False)
    service = features.groupby("service_type").agg(shipments=("shipment_id", "count"), delayed=("is_delayed", "sum"), mean_delivery_hours=("delivery_hours", "mean")).reset_index()
    service["delay_rate"] = service["delayed"] / service["shipments"]
    service.to_csv(OUT / "service_summary.csv", index=False)
    region = features.groupby("region").agg(shipments=("shipment_id", "count"), delayed=("is_delayed", "sum"), mean_delivery_hours=("delivery_hours", "mean")).reset_index()
    region["delay_rate"] = region["delayed"] / region["shipments"]
    region.to_csv(OUT / "region_summary.csv", index=False)
    summary = pd.DataFrame([{"raw_rows": len(raw), "clean_rows": len(cleaned), "duplicates_removed": len(raw)-len(cleaned), "delay_rate": features["is_delayed"].mean(), "mean_delivery_hours": features["delivery_hours"].mean(), "sentinel_weights_removed": int((pd.to_numeric(raw["parcel_weight_kg"], errors="coerce") > 100).sum())}])
    summary.to_csv(OUT / "eda_summary.csv", index=False)
    print("Part A complete. Modelling rows:", len(features))
if __name__ == "__main__": main()

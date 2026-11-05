"""Package reproducibility metadata for the trained shipment models."""
from pathlib import Path
import json, joblib, pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, calculate_impute_values
ROOT=Path(__file__).resolve().parent

def main():
    raw=pd.read_csv(ROOT/"data/raw_shipments.csv")
    impute=calculate_impute_values(clean_shipments(raw))
    pd.DataFrame({"feature":FEATURES,"order":range(len(FEATURES))}).to_csv(ROOT/"models/feature_order.csv",index=False)
    pd.DataFrame([impute]).to_csv(ROOT/"models/impute_values.csv",index=False)
    manifest={"regression_model":"models/delay_model.pkl","classification_model":"models/classification_model.pkl","features":FEATURES,"delay_threshold":0.5}
    (ROOT/"models/model_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Model metadata packaged.")
if __name__=="__main__": main()

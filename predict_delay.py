"""Score the unseen data/new_shipments.csv file with the trained models."""
from pathlib import Path
import joblib, pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES
ROOT=Path(__file__).resolve().parent

def main():
    hubs=pd.read_csv(ROOT/"data/hubs.csv")
    raw=pd.read_csv(ROOT/"data/new_shipments.csv")
    cleaned=clean_shipments(raw)
    impute=pd.read_csv(ROOT/"models/impute_values.csv").iloc[0].to_dict()
    X=build_features(cleaned,hubs,impute)[FEATURES].astype(float)
    reg=joblib.load(ROOT/"models/delay_model.pkl")
    clf=joblib.load(ROOT/"models/classification_model.pkl")
    threshold=float(pd.read_csv(ROOT/"models/delay_threshold.csv").iloc[0]["delay_probability_threshold"])
    probability=clf.predict_proba(X)[:,1]
    result=cleaned[["shipment_id","hub_id","service_type","payment_mode"]].copy()
    result["predicted_delivery_hours"]=reg.predict(X).round(2)
    result["delay_probability"]=probability.round(4)
    result["predicted_is_delayed"]=(probability>=threshold).astype(int)
    result["risk_level"]=pd.cut(probability,[-0.001,0.35,0.65,1.001],labels=["Low","Medium","High"]).astype(str)
    result.to_csv(ROOT/"outputs/new_shipment_predictions.csv",index=False)
    print(result.to_string(index=False))
if __name__=="__main__": main()

"""Part C: classify whether a shipment will exceed its promised delivery time."""
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_csv(ROOT/"data/modelling_table.csv")
    X=df[FEATURES].astype(float); y=df["is_delayed"].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1)
    model.fit(X_train,y_train); probability=model.predict_proba(X_test)[:,1]; pred=(probability>=0.5).astype(int)
    metrics={"model":"RandomForestClassifier","test_rows":len(y_test),"threshold":0.5,"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,probability) if y_test.nunique()>1 else float("nan")}
    pd.DataFrame([metrics]).to_csv(ROOT/"outputs/classification_metrics.csv",index=False)
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    pd.DataFrame([{"true_negative":tn,"false_positive":fp,"false_negative":fn,"true_positive":tp}]).to_csv(ROOT/"outputs/confusion_matrix.csv",index=False)
    pd.DataFrame({"actual_is_delayed":y_test.values,"predicted_is_delayed":pred,"delay_probability":probability}).to_csv(ROOT/"outputs/classification_predictions.csv",index=False)
    joblib.dump(model,ROOT/"models/classification_model.pkl")
    pd.DataFrame([{"delay_probability_threshold":0.5,"decision_rule":"flag as delayed when predicted probability >= threshold"}]).to_csv(ROOT/"models/delay_threshold.csv",index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
if __name__=="__main__": main()

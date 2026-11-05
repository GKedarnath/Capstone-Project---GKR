"""Part B: estimate delivery time with a reproducible regression model."""
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_csv(ROOT/"data/modelling_table.csv")
    X=df[FEATURES].astype(float); y=df["delivery_hours"].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=350,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(X_train,y_train); pred=model.predict(X_test)
    metrics={"model":"RandomForestRegressor","test_rows":len(y_test),"MAE_hours":mean_absolute_error(y_test,pred),"RMSE_hours":mean_squared_error(y_test,pred)**0.5,"R2":r2_score(y_test,pred)}
    pd.DataFrame([metrics]).to_csv(ROOT/"outputs/regression_metrics.csv",index=False)
    pd.DataFrame({"actual_delivery_hours":y_test.values,"predicted_delivery_hours":pred,"absolute_error_hours":abs(y_test.values-pred)}).to_csv(ROOT/"outputs/regression_predictions.csv",index=False)
    joblib.dump(model,ROOT/"models/delay_model.pkl")
    print(pd.DataFrame([metrics]).to_string(index=False))
if __name__=="__main__": main()

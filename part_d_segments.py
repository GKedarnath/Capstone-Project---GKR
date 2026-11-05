"""Part D: group hubs into operational segments and suggest practical actions."""
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_csv(ROOT/"data/modelling_table.csv")
    agg=df.groupby(["hub_id","hub_city","region","hub_tier"],as_index=False).agg(shipments=("shipment_id","count"),delay_rate=("is_delayed","mean"),avg_distance_km=("distance_km","mean"),avg_stops=("n_stops","mean"),avg_delivery_hours=("delivery_hours","mean"))
    cols=["delay_rate","avg_distance_km","avg_stops","avg_delivery_hours"]
    scaled=StandardScaler().fit_transform(agg[cols])
    agg["segment_id"]=KMeans(n_clusters=3,random_state=42,n_init=20).fit_predict(scaled)+1
    ranks=agg.groupby("segment_id")["delay_rate"].mean().rank(method="dense").astype(int)
    agg["segment_profile"]=agg["segment_id"].map({s:("Higher-delay watchlist" if ranks[s]==3 else "Moderate performance" if ranks[s]==2 else "Relatively stable") for s in ranks.index})
    agg.to_csv(ROOT/"outputs/hub_segments.csv",index=False)
    actions=[]
    for _,r in agg.iterrows():
        if r.delay_rate>=agg.delay_rate.quantile(.75): action="Review route capacity, dispatch cut-offs and weather contingency planning."
        elif r.avg_distance_km>=agg.avg_distance_km.quantile(.75): action="Review long-lane routing and consider intermediate hand-off planning."
        else: action="Maintain current operating plan and monitor weekly delay trends."
        actions.append({"hub_id":r.hub_id,"hub_city":r.hub_city,"segment_id":r.segment_id,"delay_rate":r.delay_rate,"recommended_action":action})
    pd.DataFrame(actions).to_csv(ROOT/"outputs/operations_recommendations.csv",index=False)
    print("Hub segments and operational recommendations saved.")
if __name__=="__main__": main()

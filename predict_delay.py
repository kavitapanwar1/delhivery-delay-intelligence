from pathlib import Path
import pandas as pd
import joblib
from src.shipment_prep import FEATURES, clean_shipments, build_features
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'data/new_shipments.csv'); hubs=pd.read_csv(ROOT/'data/hubs.csv')
med=pd.read_csv(ROOT/'models/impute_values.csv').set_index('column')['median'].to_dict()
clean,_=clean_shipments(raw,medians=med); data=build_features(clean,hubs)
order=pd.read_csv(ROOT/'models/feature_order.csv')['feature'].tolist()
model=joblib.load(ROOT/'models/delay_model.pkl'); threshold=float(pd.read_csv(ROOT/'models/delay_threshold.csv').iloc[0,0])
segments=pd.read_csv(ROOT/'outputs/hub_segments.csv')[['hub_id','segment']]
data['delay_probability']=model.predict_proba(data[order])[:,1].round(4)
data['flagged']=(data['delay_probability']>=threshold).astype(int)
data=data.merge(segments,on='hub_id',how='left',validate='many_to_one')
data['action']='No action'
data.loc[(data.flagged==1)&(data.segment=='Escalate'),'action']='Escalate to hub manager'
data.loc[(data.flagged==1)&(data.segment!='Escalate'),'action']='Proactive customer call'
result=data[['shipment_id','hub_id','segment','delay_probability','flagged','action']]
result.to_csv(ROOT/'outputs/new_shipment_actions.csv',index=False,float_format='%.4f')
print('Scored',len(result),'new shipments; flagged',int(result.flagged.sum()))

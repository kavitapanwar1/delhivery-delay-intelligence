from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments
ROOT=Path(__file__).resolve().parent; (ROOT/'models').mkdir(exist_ok=True); (ROOT/'outputs').mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
cfg=pd.read_csv(ROOT/'outputs/part_c_final_config.csv').iloc[0]
md=None if str(cfg.max_depth)=='unlimited' else int(cfg.max_depth)
model=Pipeline([('selector',SelectKBest(f_classif,k=int(cfg.selector_k))),('model',RandomForestClassifier(n_estimators=int(cfg.n_estimators),max_depth=md,min_samples_leaf=int(cfg.min_samples_leaf),random_state=42))])
model.fit(X_train,y_train)
joblib.dump(model,ROOT/'models/delay_model.pkl')
pd.DataFrame({'threshold':[float(cfg.threshold)]}).to_csv(ROOT/'models/delay_threshold.csv',index=False)
pd.DataFrame({'feature':FEATURES}).to_csv(ROOT/'models/feature_order.csv',index=False)
raw=pd.read_csv(ROOT/'data/raw_shipments.csv'); _,medians=clean_shipments(raw)
pd.DataFrame([{'column':k,'median':v} for k,v in medians.items()]).to_csv(ROOT/'models/impute_values.csv',index=False)
loaded=joblib.load(ROOT/'models/delay_model.pkl')
p1=model.predict_proba(X_test)[:,1]; p2=loaded.predict_proba(X_test)[:,1]
pd.DataFrame([{'metric':'test_rows_scored','value':len(X_test)},{'metric':'reload_max_abs_probability_diff','value':np.max(np.abs(p1-p2))},{'metric':'threshold','value':float(cfg.threshold)}]).round(10).to_csv(ROOT/'outputs/part_d_reload_check.csv',index=False)
print('Model packaged; reload max probability difference:',np.max(np.abs(p1-p2)))

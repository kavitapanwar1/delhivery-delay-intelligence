from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
ROOT=Path(__file__).resolve().parent
out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv')
X=df[FEATURES]; y=df.delivery_hours
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42)
scaler=StandardScaler().fit(X_train)
tr_scaled=scaler.transform(X_train); te_scaled=scaler.transform(X_test)
pd.DataFrame([('train_rows',len(X_train)),('val_rows',len(X_val)),('test_rows',len(X_test)),('train_scaled_max_abs_mean',np.abs(tr_scaled.mean(axis=0)).max()),('test_scaled_max_abs_mean',np.abs(te_scaled.mean(axis=0)).max())],columns=['metric','value']).round(4).to_csv(out/'part_b_checks.csv',index=False)
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
depths=[1,2,3,4,6,8,12,None]; trees={}; diag=[]
for d in depths:
 m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train); trees[d]=m
 diag.append({'max_depth':'unlimited' if d is None else d,'train_rmse':rmse(y_train,m.predict(X_train)),'val_rmse':rmse(y_val,m.predict(X_val))})
pd.DataFrame(diag).round(4).to_csv(out/'part_b_tree_depth.csv',index=False)
best_depth=min(depths,key=lambda d:(rmse(y_val,trees[d].predict(X_val)),999 if d is None else d))
models={
 'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'decision_tree':trees[best_depth],
 'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),
 'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42),
}
def scores(name,split,yt,yp):
 mse=mean_squared_error(yt,yp)
 return {'model':name,'split':split,'MAE':mean_absolute_error(yt,yp),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(yt,yp)}
rows=[scores('mean_baseline','val',y_val,np.full(len(y_val),y_train.mean()))]
for name,m in models.items():
 m.fit(X_train,y_train); rows.append(scores(name,'val',y_val,m.predict(X_val)))
learning=[]
for lr in [.01,.1,1.0]:
 m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
 p=m.predict(X_val); learning.append({'learning_rate':lr,'val_rmse':rmse(y_val,p),'val_r2':r2_score(y_val,p)})
pd.DataFrame(learning).round(4).to_csv(out/'part_b_gb_learning_rate.csv',index=False)
ranked=sorted(models,key=lambda n:next(r['RMSE'] for r in rows if r['model']==n))
cv_rows=[]; cv=KFold(n_splits=5,shuffle=True,random_state=42)
X_tv=pd.concat([X_train,X_val]); y_tv=pd.concat([y_train,y_val])
for name in ranked[:2]:
 m=models[name]
 vals=cross_val_score(m,X_tv,y_tv,cv=cv)
 cv_rows.append({'model':name,'mean_r2':vals.mean(),'std_r2':vals.std()})
pd.DataFrame(cv_rows).round(4).to_csv(out/'part_b_cv.csv',index=False)
winner=ranked[0]; rows.append(scores(winner,'test',y_test,models[winner].predict(X_test)))
pd.DataFrame(rows,columns=['model','split','MAE','MSE','RMSE','R2']).round(4).to_csv(out/'part_b_metrics.csv',index=False)
print('Part B complete. Validation winner:',winner,'depth:',best_depth)

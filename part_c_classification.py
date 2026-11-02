from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS
ROOT=Path(__file__).resolve().parent; out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
def metric_row(name,yt,yp):
 tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
 return {'model':name,'TN':tn,'FP':fp,'FN':fn,'TP':tp,'accuracy':accuracy_score(yt,yp),'precision':precision_score(yt,yp,zero_division=0),'recall':recall_score(yt,yp,zero_division=0),'f1':f1_score(yt,yp,zero_division=0)}
rows=[metric_row('blind_all_on_time',y_val,np.zeros(len(y_val),dtype=int))]
log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())]).fit(X_train,y_train)
rows.append(metric_row('logistic_regression',y_val,log.predict(X_val)))
tree_scores=[]; trees={}
for d in [2,3,4,5,6,8,10]:
 m=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(X_train,y_train); trees[d]=m
 f=f1_score(y_val,m.predict(X_val),zero_division=0); tree_scores.append((d,f))
best_depth=min(tree_scores,key=lambda z:(-z[1],z[0]))[0]
rows.append(metric_row(f'decision_tree_depth_{best_depth}',y_val,trees[best_depth].predict(X_val)))
rf_all=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
rows.append(metric_row('random_forest_default',y_val,rf_all.predict(X_val)))
pd.DataFrame(rows).round(4).to_csv(out/'part_c_models.csv',index=False)
# Feature signal scores, fitted on training data only.
selector_all=SelectKBest(f_classif,k='all').fit(X_train,y_train)
f_scores=selector_all.scores_
importance=rf_all.feature_importances_
order=np.argsort(-f_scores,kind='stable')
ranks=np.empty(len(FEATURES),dtype=int); ranks[order]=np.arange(1,len(FEATURES)+1)
pd.DataFrame({'feature':FEATURES,'f_score':f_scores,'f_rank':ranks,'rf_importance':importance,'is_noise':[f in NOISE_COLUMNS for f in FEATURES]}).sort_values('f_rank').round(4).to_csv(out/'part_c_feature_scores.csv',index=False)
select_rows=[]
for k in [5,7,10,12,17]:
 p=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
 select_rows.append({'k':k,'val_f1':f1_score(y_val,p.predict(X_val),zero_division=0)})
pd.DataFrame(select_rows).round(4).to_csv(out/'part_c_selectk.csv',index=False)
# k=7 selector is fitted only on training rows; reuse its transformed matrices for tuning.
selector=SelectKBest(f_classif,k=7).fit(X_train,y_train)
Xt=selector.transform(X_train); Xv=selector.transform(X_val)
param_grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid=param_grid,cv=5,n_jobs=-1)
grid.fit(Xt,y_train)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),param_distributions=param_grid,n_iter=9,cv=5,random_state=42,n_jobs=-1)
rand.fit(Xt,y_train)
def tune_row(name,candidates,folds,fits,best_params,best_cv,model):
 return {'search':name,'candidates':candidates,'cv_folds':folds,'total_fits':fits,'best_params':best_params,'best_cv_accuracy':best_cv,'val_f1':f1_score(y_val,model.predict(Xv),zero_division=0)}
def fmt_params(p):
 return str(p).replace("'", '"')
deforest=RandomForestClassifier(n_estimators=100,random_state=42,min_samples_leaf=1).fit(Xt,y_train)
tuning=[tune_row('grid_search',27,5,135,fmt_params(grid.best_params_),grid.best_score_,grid.best_estimator_),
 tune_row('randomized_search',9,5,45,fmt_params(rand.best_params_),rand.best_score_,rand.best_estimator_),
 tune_row('default_forest_same_7_features',1,0,1,'defaults',np.nan,deforest)]
pd.DataFrame(tuning).round(4).to_csv(out/'part_c_tuning.csv',index=False)
# Highest validation F1 wins; ties favor default, then grid, then randomized.
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen=sorted(tuning,key=lambda r:(-r['val_f1'],priority[r['search']]))[0]
if chosen['search']=='grid_search': params=grid.best_params_.copy()
elif chosen['search']=='randomized_search': params=rand.best_params_.copy()
else: params={'n_estimators':100,'max_depth':None,'min_samples_leaf':1}
final_model=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**params))]).fit(X_train,y_train)
probs=final_model.predict_proba(X_val)[:,1]
thresholds=[]
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
 thresholds.append({'threshold':t,**{k:v for k,v in metric_row('tmp',y_val,(probs>=t).astype(int)).items() if k!='model'}})
pd.DataFrame(thresholds).round(4).to_csv(out/'part_c_thresholds.csv',index=False)
valid=[r for r in thresholds if r['recall']>=.80]
threshold=max((r['threshold'] for r in valid),default=.1)
test_probs=final_model.predict_proba(X_test)[:,1]
testrow=metric_row('final',y_test,(test_probs>=threshold).astype(int))
pd.DataFrame([{k:v for k,v in testrow.items() if k!='model'}|{'threshold':threshold}],columns=['threshold','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4).to_csv(out/'part_c_final_test.csv',index=False)
pd.DataFrame([{'selector_k':7,'n_estimators':params['n_estimators'],'max_depth':'unlimited' if params.get('max_depth') is None else params.get('max_depth'),'min_samples_leaf':params['min_samples_leaf'],'threshold':threshold,'chosen_from':chosen['search']}]).to_csv(out/'part_c_final_config.csv',index=False)
print('Part C complete. Chosen:',chosen['search'],'threshold:',threshold,'split delayed:',int(y_train.sum()),int(y_val.sum()),int(y_test.sum()))

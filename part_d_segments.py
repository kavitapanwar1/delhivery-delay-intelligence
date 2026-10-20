from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
ROOT=Path(__file__).resolve().parent; out=ROOT/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv')
profile=df.groupby('hub_id',sort=True).agg(shipments=('shipment_id','count'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).reset_index()
profile.to_csv(out/'hub_profile.csv',index=False,float_format='%.4f')
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops']
scaled=StandardScaler().fit_transform(profile[cols])
sep=[]
for k in [2,3,4,5]:
 km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(scaled); labels=km.labels_
 dist=pairwise_distances(scaled); same=[]; across=[]
 for i in range(len(profile)):
  for j in range(i+1,len(profile)):
   (same if labels[i]==labels[j] else across).append(dist[i,j])
 sizes=sorted(np.bincount(labels).tolist(),reverse=True)
 sep.append({'K':k,'across_over_within':np.mean(across)/np.mean(same),'cluster_sizes':'-'.join(map(str,sizes))})
pd.DataFrame(sep).round(4).to_csv(out/'part_d_separation.csv',index=False)
km=KMeans(n_clusters=3,random_state=42,n_init=10).fit(scaled)
profile['cluster']=km.labels_
cluster_order=profile.groupby('cluster')['delay_rate'].mean().sort_values().index.tolist()
name_map={cluster_order[0]:'Standard',cluster_order[1]:'Watch',cluster_order[2]:'Escalate'}
profile['segment']=profile['cluster'].map(name_map)
hubs=pd.read_csv(ROOT/'data/hubs.csv')
hub_segments=profile[['hub_id','segment']].merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id',how='left')
hub_segments.to_csv(out/'hub_segments.csv',index=False)
segprof=profile.groupby('segment').agg(hubs=('hub_id','count'),mean_delivery_hours=('mean_delivery_hours','mean'),delay_rate=('delay_rate','mean'),mean_distance_km=('mean_distance_km','mean'),mean_n_stops=('mean_n_stops','mean')).reset_index()
segprof['_order']=segprof.segment.map({'Standard':0,'Watch':1,'Escalate':2}); segprof.sort_values('_order').drop(columns='_order').to_csv(out/'segment_profile.csv',index=False,float_format='%.4f')
expected={1:'Standard',2:'Watch',3:'Escalate'}
mismatch=hub_segments[hub_segments.apply(lambda r:r.segment!=expected[r.hub_tier],axis=1)]
mismatch.to_csv(out/'tier_vs_segment_mismatch.csv',index=False)
print('Part D segments complete:',segprof[['segment','hubs']].to_dict('records'))

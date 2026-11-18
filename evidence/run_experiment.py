"""Reproducible CWRU 48 kHz DE benchmark. No test-set tuning."""
from pathlib import Path
from urllib.request import urlopen
import json,hashlib,time,platform,csv,sys,sqlite3,warnings
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,precision_score,recall_score,f1_score
import sklearn
ROOT=Path(__file__).parent;DATA=ROOT/'data';OUT=ROOT/'results';DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
SEED=42;WINDOW=2048
# Defined before fitting. Each entire recording belongs to exactly one split.
records=[(97,'Normal_0',0,0,'train'),(98,'Normal_1',1,0,'validation'),(99,'Normal_2',2,0,'test'),(100,'Normal_3',3,0,'test'),(110,'IR007_1',1,1,'validation'),(111,'IR007_2',2,1,'test'),(112,'IR007_3',3,1,'test'),(123,'B007_1',1,1,'validation'),(124,'B007_2',2,1,'test'),(125,'B007_3',3,1,'test'),(136,'OR007@6_1',1,1,'validation'),(137,'OR007@6_2',2,1,'test'),(138,'OR007@6_3',3,1,'test')]
def features(w):
 # Raw acceleration amplitudes retained; kurtosis is Pearson population kurtosis.
 rms=np.sqrt(np.mean(w*w,axis=1));center=w-w.mean(axis=1,keepdims=True);v=np.mean(center**2,axis=1)
 kurt=np.mean(center**4,axis=1)/(v*v);crest=np.max(np.abs(w),axis=1)/rms
 return np.column_stack([rms,kurt,crest])
rows=[];manifest=[];samples=[];seen=set()
for fid,name,load,label,split in records:
 path=DATA/f'{fid}.mat';url=f'https://engineering.case.edu/sites/default/files/{fid}.mat'
 if not path.exists():
  with urlopen(url,timeout=90) as r:path.write_bytes(r.read())
 digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest not in seen;seen.add(digest)
 expected_path=ROOT/'expected_manifest.json'
 if expected_path.exists():
  expected={r['id']:r['sha256'] for r in json.loads(expected_path.read_text())};assert digest==expected[str(fid)],'Source checksum changed: '+str(fid)
 mat=loadmat(path);keys=[f'X{fid:03d}_DE_time'];assert keys[0] in mat,(fid,list(mat))
 signal=np.asarray(mat[keys[0]],dtype=float).ravel();assert np.isfinite(signal).all()
 n=len(signal)//WINDOW;w=signal[:n*WINDOW].reshape(n,WINDOW);f=features(w);assert np.isfinite(f).all()
 manifest.append(dict(id=str(fid),name=name,load_hp=load,label=label,split=split,url=url,sha256=digest,bytes=path.stat().st_size,channel=keys[0],samples=len(signal),windows=n,discarded_tail_samples=len(signal)%WINDOW,sampling_hz=48000,fault_diameter_in=0 if label==0 else .007))
 for j,a in enumerate(f):rows.append(dict(recording=str(fid),name=name,split=split,label=label,window=j,rms=float(a[0]),kurtosis=float(a[1]),crest_factor=float(a[2])))
 if split=='test':samples.append(dict(id=str(fid),name=name,label=label,load_hp=load,window=0,samples=w[0].tolist(),features=f[0].tolist(),rms_trend=f[:,0].tolist(),window_count=n))
 print(name,split,n,flush=True)
X=np.array([[r['rms'],r['kurtosis'],r['crest_factor']] for r in rows]);y=np.array([r['label'] for r in rows]);parts=np.array([r['split'] for r in rows]);tr=parts=='train';va=parts=='validation';te=parts=='test';assert set(y[tr])=={0}
scaler=StandardScaler().fit(X[tr]);Z=scaler.transform(X)
iso=IsolationForest(n_estimators=100,max_samples='auto',random_state=SEED,n_jobs=1).fit(Z[tr])
ae=MLPRegressor(hidden_layer_sizes=(6,2,6),activation='tanh',solver='lbfgs',alpha=.001,max_iter=20000,max_fun=300000,tol=1e-7,random_state=SEED)
with warnings.catch_warnings(record=True) as ws:
 warnings.simplefilter('always');ae.fit(Z[tr],Z[tr]);fit_warnings=[str(w.message) for w in ws]
scorers={'RMS threshold':lambda A:X[A,0], 'Isolation Forest':lambda A:-iso.score_samples(Z[A]), 'Autoencoder':lambda A:np.mean((ae.predict(Z[A])-Z[A])**2,axis=1)}
def metrics(y,s,t):
 pred=(s>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(roc_auc=float(roc_auc_score(y,s)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),recall=float(recall_score(y,pred,zero_division=0)),precision=float(precision_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),false_alarm_rate=float(fp/(fp+tn)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],n=len(y),healthy=int(sum(y==0)),fault=int(sum(y==1)))
models=[];scores={};thresholds={}
for name,func in scorers.items():
 s=func(va);unique=np.unique(s);candidates=np.r_[np.nextafter(unique[0],-np.inf),(unique[:-1]+unique[1:])/2,np.nextafter(unique[-1],np.inf)]
 # Threshold selected only using validation balanced accuracy. Ties prefer lower false-alarm rate.
 threshold=max(candidates,key=lambda t:(balanced_accuracy_score(y[va],s>=t),-np.mean((s[y[va]==0]>=t))))
 thresholds[name]=float(threshold);scores[name]=func(np.ones(len(y),dtype=bool));models.append(dict(name=name,threshold=float(threshold),validation=metrics(y[va],s,threshold)))
# Select on validation before looking at test metrics. Simpler RMS wins exact ties.
winner=max(range(3),key=lambda i:(models[i]['validation']['balanced_accuracy'],models[i]['validation']['roc_auc'],-i));selected=models[winner]['name']
for m in models:
 m['test']=metrics(y[te],scores[m['name']][te],m['threshold']);m['per_test_recording']=[]
 for rec in manifest:
  if rec['split']!='test':continue
  idx=np.array([r['recording']==rec['id'] for r in rows]);pred=scores[m['name']][idx]>=m['threshold'];m['per_test_recording'].append(dict(id=rec['id'],name=rec['name'],label=rec['label'],windows=int(idx.sum()),flagged=int(pred.sum()),flagged_rate=float(pred.mean())))
# Honest local CPU benchmark, preprocessing + selected model, no HTTP or upload.
def one(w):
 f=features(w.reshape(1,-1))
 if selected=='RMS threshold':return f[0,0]
 z=scaler.transform(f)
 return -iso.score_samples(z)[0] if selected=='Isolation Forest' else float(np.mean((ae.predict(z)-z)**2))
window=np.array(samples[0]['samples']);[one(window) for _ in range(10)];times=[]
for _ in range(100):
 t=time.perf_counter();one(window);times.append((time.perf_counter()-t)*1000)
artifact=dict(version='cwru-48k-v1',selected=selected,threshold=thresholds[selected],features=['rms','kurtosis','crest_factor'],window_size=WINDOW,sampling_hz=48000,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),autoencoder=dict(weights=[a.tolist() for a in ae.coefs_],biases=[a.tolist() for a in ae.intercepts_]),thresholds=thresholds)
# Export Isolation Forest trees with sklearn's average path-length approximation.
def avg(n):return 0. if n<=1 else 1. if n==2 else 2*(np.log(n-1)+np.euler_gamma)-2*(n-1)/n
artifact['isolation_forest']=dict(max_samples=int(iso.max_samples_),normalizer=float(avg(iso.max_samples_)),trees=[dict(left=t.tree_.children_left.tolist(),right=t.tree_.children_right.tolist(),feature=t.tree_.feature.tolist(),threshold=t.tree_.threshold.tolist(),leaf_c=[float(avg(n)) for n in t.tree_.n_node_samples]) for t in iso.estimators_])
for sample in samples:
 idx=next(i for i,r in enumerate(rows) if r['recording']==sample['id'] and r['window']==0)
 sample['scores']={k:float(v[idx]) for k,v in scores.items()};sample['score']=sample['scores'][selected];sample['prediction']='anomaly' if sample['score']>=artifact['threshold'] else 'normal'
report=dict(experiment_id='cwru-48k-v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed=SEED,dataset='CWRU drive-end, 48 kHz, 0.007-inch seeded faults, outer race at 6 oclock',window_size=WINDOW,window_ms=WINDOW/48000*1000,features=artifact['features'],split_policy='Train: Normal_0. Validation: all four selected conditions at 1 HP. Test: all four conditions at 2 and 3 HP. Entire recordings kept separate; no overlapping windows.',selection_policy='Maximum validation balanced accuracy, then validation ROC-AUC, then simpler model. No test-set tuning.',selected=selected,counts={p:dict(recordings=sum(r['split']==p for r in manifest),windows=int(sum(parts==p))) for p in ['train','validation','test']},models=models,latency=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),requests=100,scope='Local Python feature extraction + selected model; excludes network',hardware=platform.machine(),processor=platform.processor()),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),fit_warnings=fit_warnings,limitations=['Only 13 recordings and one CWRU rig; windows within a recording are correlated.','Test loads differ, but fault specimens/types are shared; this is not a new-bearing or new-factory generalisation test.','Only 0.007-inch seeded faults; no fault progression, RUL, or early-warning lead time is measured.','NASA IMS was identified in planning, but not trained or evaluated in this experiment.','Three features were chosen for simplicity, not proved optimal by ablation.','Reported window-level scores should not be treated as independent repeated trials or plant reliability.'])
for name,obj in [('manifest.json',manifest),('report.json',report),('model.json',artifact),('samples.json',samples)]: (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
with open(OUT/'features.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with open(OUT/'predictions.csv','w') as f:
 w=csv.writer(f);w.writerow(['recording','window','split','label','rms_score','isolation_forest_score','autoencoder_score','selected_prediction'])
 for i,r in enumerate(rows):w.writerow([r['recording'],r['window'],r['split'],r['label'],*[float(scores[k][i]) for k in scorers],int(scores[selected][i]>=artifact['threshold'])])
print(json.dumps({'selected':selected,'models':models,'counts':report['counts'],'latency':report['latency'],'warnings':fit_warnings},indent=2))

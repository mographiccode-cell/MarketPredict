import sys,json,joblib,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score,f1_score,precision_score,recall_score,matthews_corrcoef,average_precision_score,brier_score_loss,confusion_matrix
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
OBJ=sys.argv[1]; BASE=Path('/mnt/data')
DF=pd.read_csv(BASE/'saudi_marketing_campaigns_v5_prelaunch.csv',parse_dates=['Start_Date'])
DF['Start_Month']=DF.Start_Date.dt.month.astype(str); DF['Start_Quarter']=DF.Start_Date.dt.quarter.astype(str); DF['Start_DayOfWeek']=DF.Start_Date.dt.dayofweek.astype(str)
D=DF[(DF.Campaign_Objective==OBJ)&(DF.Split_Role=='DEVELOPMENT')].sort_values('Start_Date').copy()
exclude={'Campaign_ID','Brand_ID','Start_Date','Success','Split_Role','Campaign_Objective'}
FEATURES=[c for c in DF.columns if c not in exclude]
# objective-wise remove all-null and constants in development to reduce computation and prevent useless features
FEATURES=[c for c in FEATURES if D[c].notna().any() and D[c].nunique(dropna=True)>1]
LGB={'n_estimators':165,'learning_rate':.040,'num_leaves':15,'min_child_samples':140,'reg_lambda':16,'reg_alpha':3.5,'colsample_bytree':.74}
CAT={'iterations':150,'depth':5,'learning_rate':.040,'l2_leaf_reg':14,'random_strength':1.7}
def make_lgb(X):
 num=X.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in X.columns if c not in num]
 prep=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
 return Pipeline([('prep',prep),('model',LGBMClassifier(**LGB,subsample=.9,random_state=42,n_jobs=2,verbosity=-1))])
def fit_cat(X,y):
 Z=X.copy(); cat=[c for c in Z.columns if Z[c].dtype=='object']; num=[c for c in Z.columns if c not in cat]
 med={c:(float(Z[c].median()) if Z[c].notna().any() else 0.0) for c in num}
 for c in num: Z[c]=Z[c].fillna(med[c]).astype(float)
 for c in cat: Z[c]=Z[c].fillna('__MISSING__').astype(str)
 m=CatBoostClassifier(**CAT,loss_function='Logloss',eval_metric='AUC',verbose=False,allow_writing_files=False,random_seed=42,thread_count=2)
 m.fit(Z,y,cat_features=cat)
 return {'model':m,'cat':cat,'num':num,'med':med}
def pred_cat(b,X):
 Z=X.copy()
 for c in b['num']: Z[c]=Z[c].fillna(b['med'][c]).astype(float)
 for c in b['cat']: Z[c]=Z[c].fillna('__MISSING__').astype(str)
 return b['model'].predict_proba(Z)[:,1]
def auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
def ece(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p);e=np.linspace(0,1,bins+1);z=0.0
 for i in range(bins):
  m=(p>=e[i])&(p<(e[i+1] if i<bins-1 else e[i+1]+1e-12))
  if m.sum(): z += m.mean()*abs(p[m].mean()-y[m].mean())
 return float(z)
def metrics(T,fl,fc,w,cal,thr):
 if len(T)==0:return {'n':0}
 pl=fl.predict_proba(T[FEATURES])[:,1];pc=pred_cat(fc,T[FEATURES]);raw=w*pl+(1-w)*pc;pr=cal.predict_proba(raw.reshape(-1,1))[:,1];yp=(pr>=thr).astype(int);y=T.Success.astype(int).to_numpy()
 return {'n':int(len(T)),'brands':int(T.Brand_ID.nunique()),'success_rate':float(y.mean()),'roc_auc':auc(y,pr),'pr_auc':float(average_precision_score(y,pr)),'accuracy':float(accuracy_score(y,yp)),'balanced_accuracy':float(balanced_accuracy_score(y,yp)),'precision':float(precision_score(y,yp,zero_division=0)),'recall':float(recall_score(y,yp,zero_division=0)),'f1':float(f1_score(y,yp,zero_division=0)),'mcc':float(matthews_corrcoef(y,yp)),'brier':float(brier_score_loss(y,pr)),'ece10':ece(y,pr),'confusion_matrix':confusion_matrix(y,yp).tolist()}
# Rolling chronological folds; no random test split
folds=[('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'),('2026Q1','2026-01-01','2026-04-01')]
O=[];foldstats=[]
for name,vs,ve in folds:
 tr=D[D.Start_Date<pd.Timestamp(vs)]; va=D[(D.Start_Date>=pd.Timestamp(vs))&(D.Start_Date<pd.Timestamp(ve))]
 if len(tr)<300 or len(va)<80:continue
 l=make_lgb(tr[FEATURES]);l.fit(tr[FEATURES],tr.Success.astype(int));pl=l.predict_proba(va[FEATURES])[:,1]
 c=fit_cat(tr[FEATURES],tr.Success.astype(int));pc=pred_cat(c,va[FEATURES])
 O.append(pd.DataFrame({'y':va.Success.astype(int).to_numpy(),'lgb':pl,'cat':pc,'fold':name}))
 foldstats.append({'fold':name,'train_n':len(tr),'val_n':len(va),'lgb_auc':auc(va.Success,pl),'cat_auc':auc(va.Success,pc)})
OOF=pd.concat(O,ignore_index=True)
# Ensemble weight selected exclusively on rolling OOF; tie favors 50/50
W=np.linspace(0,1,21); scores=[]
for w in W:scores.append((auc(OOF.y,w*OOF.lgb+(1-w)*OOF.cat),float(w)))
best=max(x[0] for x in scores); eligible=[x for x in scores if x[0]>=best-0.0005]; w=min(eligible,key=lambda x:abs(x[1]-.5))[1]
raw=w*OOF.lgb.to_numpy()+(1-w)*OOF.cat.to_numpy()
# Calibration and threshold from rolling OOF only
cal=LogisticRegression(C=10,solver='lbfgs');cal.fit(raw.reshape(-1,1),OOF.y);cp=cal.predict_proba(raw.reshape(-1,1))[:,1]
ths=np.linspace(.25,.75,101);bs=[balanced_accuracy_score(OOF.y,(cp>=t).astype(int)) for t in ths];thr=float(ths[int(np.argmax(bs))])
# Final model frozen using DEVELOPMENT only. No group/temporal/lockbox labels used in fit.
fl=make_lgb(D[FEATURES]);fl.fit(D[FEATURES],D.Success.astype(int));fc=fit_cat(D[FEATURES],D.Success.astype(int))
trainraw=w*fl.predict_proba(D[FEATURES])[:,1]+(1-w)*pred_cat(fc,D[FEATURES]); trainauc=auc(D.Success,trainraw)
g=DF[(DF.Campaign_Objective==OBJ)&(DF.Split_Role=='GROUP_VALIDATION_UNSEEN_BRANDS')].copy(); t=DF[(DF.Campaign_Objective==OBJ)&(DF.Split_Role=='TEMPORAL_HOLDOUT_SEEN_BRANDS_V5')].copy()
GM=metrics(g,fl,fc,w,cal,thr);TM=metrics(t,fl,fc,w,cal,thr)
bundle={'dataset_version':'V5','objective':OBJ,'features':FEATURES,'lgb':fl,'cat':fc,'weight_lgb':w,'calibrator':cal,'threshold':thr,'lgb_config':LGB,'cat_config':CAT,'sealed_lockbox_evaluated':False}
slug=OBJ.replace(' ','_').lower();joblib.dump(bundle,BASE/f'v5_model_{slug}.joblib')
M={'dataset_version':'V5','objective':OBJ,'development_n':int(len(D)),'rolling_oof_n':int(len(OOF)),'rolling_oof_auc_raw':auc(OOF.y,raw),'rolling_oof_auc_calibrated':auc(OOF.y,cp),'rolling_oof_brier_raw':float(brier_score_loss(OOF.y,raw)),'rolling_oof_brier_calibrated':float(brier_score_loss(OOF.y,cp)),'rolling_oof_ece_raw':ece(OOF.y,raw),'rolling_oof_ece_calibrated':ece(OOF.y,cp),'train_auc_final':trainauc,'train_oof_gap':float(trainauc-auc(OOF.y,raw)),'weight_lgb':w,'threshold':thr,'rolling_folds':foldstats,'group_validation_unseen_brands':GM,'temporal_holdout_seen_brands_2026':TM,'final_lockbox_unseen_brands_2026':'SEALED_NOT_EVALUATED'}
(BASE/f'v5_metrics_{slug}.json').write_text(json.dumps(M,indent=2),encoding='utf-8')
print(json.dumps({'objective':OBJ,'features':len(FEATURES),'dev_n':len(D),'oof_auc':M['rolling_oof_auc_raw'],'gap':M['train_oof_gap'],'w_lgb':w,'group_auc':GM.get('roc_auc'),'group_bal':GM.get('balanced_accuracy'),'temporal_auc':TM.get('roc_auc'),'temporal_bal':TM.get('balanced_accuracy')},indent=2))

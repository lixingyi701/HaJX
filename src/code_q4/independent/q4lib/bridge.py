import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import least_squares
from .common import *

COLS=dict(zip(['LB_IFEval','LB_BBH','LB_MATH','LB_GPQA','LB_MUSR','LB_MMLU_PRO'],DIMS))

def fit_map(x,y,kind):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if kind=='constant':return [float(y.mean())]
    if kind=='linear':
        o=least_squares(lambda p:np.clip(p[0]-p[1]*x,0,100)-y,[float(y.mean()),.01],bounds=([-np.inf,0],[np.inf,np.inf]))
    else:
        o=least_squares(lambda p:100*expit(p[0]-p[1]*x)-y,[logit(np.clip(y.mean()/100,.001,.999)),.1],bounds=([-np.inf,0],[np.inf,np.inf]),max_nfev=2000)
    if not o.success:raise ValueError('bridge optimizer failed')
    return o.x.tolist()

def map_predict(p,x,kind):
    x=np.asarray(x,float)
    if kind=='constant':return np.full_like(x,p[0])
    if kind=='linear':return np.clip(p[0]-p[1]*x,0,100)
    return 100*expit(p[0]-p[1]*x)

def run(data,out,cfg,rng,models):
    c5=pd.read_csv(data/'loss_benchmark_bridge.csv');c6=pd.read_csv(data/'loss_benchmark_bridge_expanded.csv')
    allb=pd.concat([c6.assign(input='C6'),c5.assign(input='C5')],ignore_index=True)
    keys=['Model','Loss_Source','Val_Loss']
    duplicates=allb[allb.duplicated(keys,keep=False)];csv(out/'tables/bridge_duplicate_rows.csv',duplicates)
    b=allb.drop_duplicates(keys).copy();b['family_id']=b.Model.map(family)
    b['tier']=b.Loss_Comparability.str.extract(r'^(High|Medium|Low)',expand=False)
    b['model_type']=b.Model.map(models.set_index('model_id').model_type).fillna('unknown')
    csv(out/'tables/bridge_input.csv',b)
    groups=[('HIGH_STRICT',b[b.tier=='High'])]+[('SOURCE:'+str(s),g) for s,g in b.groupby('Loss_Source')]+[('POOLED_EXPLORATORY_UNCALIBRATED',b)]
    fits=[];validation=[];predrows=[];drawrows=[]
    for scope,g0 in groups:
        for col,dim in COLS.items():
            g=g0.dropna(subset=['Val_Loss',col]);status='FITTED_EXPLORATORY' if 'POOLED' in scope else 'FITTED_WITHIN_SOURCE'
            meta=dict(scope=scope,dimension=dim,n=len(g),families=g.family_id.nunique(),
                loss_support=[g.Val_Loss.min(),g.Val_Loss.max()],score_scale='C6 reported leaderboard score; normalization not independently cross-calibrated',calibration_status='BRIDGE_UNCALIBRATED')
            if len(g)<10 or g.family_id.nunique()<3:
                fits.append(dict(**meta,status='INSUFFICIENT_CROSS_FAMILY_SUPPORT'));continue
            scores={}
            for form in ['constant','linear','sigmoid']:
                errs=[]
                for fam in g.family_id.unique():
                    tr=g[g.family_id!=fam];te=g[g.family_id==fam]
                    try:
                        p=fit_map(tr.Val_Loss,tr[col],form);pr=map_predict(p,te.Val_Loss,form);er=pr-te[col].to_numpy()
                        errs.extend(er.tolist());validation.append(dict(scope=scope,dimension=dim,form=form,heldout_family=fam,n=len(te),MAE=np.abs(er).mean(),RMSE=np.sqrt(np.mean(er**2))))
                    except ValueError:pass
                scores[form]=np.mean(np.abs(errs)) if len(errs)==len(g) else np.inf
            best=min(scores,key=scores.get)
            # Do not prefer a nonlinear model for a negligible (<0.1 point) error gain.
            if scores['constant']<=scores[best]+.1:best='constant'
            p=fit_map(g.Val_Loss,g[col],best)
            fits.append(dict(**meta,status=status,form=best,parameters=p,CV_MAE=scores,beta_interpretation='nonnegative: lower loss predicts no worse score'))
            for row,pred in zip(g.itertuples(),map_predict(p,g.Val_Loss,best)):
                predrows.append(dict(scope=scope,dimension=dim,model_id=row.Model,loss=row.Val_Loss,observed=getattr(row,col),prediction=pred))
            grid=np.linspace(g.Val_Loss.min(),g.Val_Loss.max(),60);draws=[]
            for bi in range(cfg['bootstrap']):
                bs=resample_clusters(g,rng)
                try:draws.append(map_predict(fit_map(bs.Val_Loss,bs[col],best),grid,best))
                except ValueError:continue
            if draws:
                qq=np.quantile(draws,[.05,.25,.5,.75,.95],axis=0)
                for i,x in enumerate(grid):drawrows.append(dict(scope=scope,dimension=dim,loss=x,form=best,p05=qq[0,i],p25=qq[1,i],p50=qq[2,i],p75=qq[3,i],p95=qq[4,i],bootstrap_valid=len(draws),uncertainty_scope='within_fit_cluster_curve_only; cross_source_calibration_error_unknown'))
    dump(out/'interface/P4_bridge_params.json',{'status':'BRIDGE_UNCALIBRATED','fits':fits,
        'resource_predictions_enabled':False,'reason':'No paired Q2 loss and same-scale C loss; Medium weights do not establish scale equivalence',
        'joint_residual_rule':'Resample whole model rows across all six dimensions; no independent dimension residual sampling',
        'propagation_status':'Local curve bootstrap available; upstream calibration uncertainty cannot be quantified without overlap'})
    csv(out/'tables/bridge_validation.csv',validation,columns=None if validation else ['scope','status'])
    csv(out/'tables/bridge_fitted.csv',predrows,columns=None if predrows else ['scope','dimension','loss','observed','prediction'])
    csv(out/'tables/bridge_curve_intervals.csv',drawrows,columns=None if drawrows else ['scope','status'])
    return fits

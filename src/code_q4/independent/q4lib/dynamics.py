import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import least_squares,minimize
from scipy.integrate import quad
from .common import *

FORMS={'intercept':[0],'scale':[0,1],'time':[0,2],'joint':[0,1,2]}

def fit(d,form='joint',tau=None,ref=None):
    if len(d)<3:raise ValueError('too few rows')
    ref=ref or {'logc':float(np.log(d.compute_flops).median()),'date':str(pd.to_datetime(d.date).min())}
    X=np.c_[np.ones(len(d)),np.log(d.compute_flops)-ref['logc'],years(d.date,ref['date'])]
    ids=FORMS[form];A=X[:,ids];y=d.average_official.to_numpy(float)
    w=(1/d.groupby('family_id').family_id.transform('size')).to_numpy();w=w/w.mean()
    start=np.zeros(len(ids));start[0]=logit(np.clip(np.average(y,weights=w)/100,.001,.999))
    if 1 in ids:start[ids.index(1)]=.1
    lo=np.array([0 if i==1 else -np.inf for i in ids]);hi=np.full(len(ids),np.inf)
    if tau is None:
        opt=least_squares(lambda v:np.sqrt(w)*(100*expit(A@v)-y),start,bounds=(lo,hi),loss='soft_l1',f_scale=5,max_nfev=2500)
    else:
        def loss(v):
            u=y-100*expit(A@v)
            return np.average(np.maximum(tau*u,(tau-1)*u),weights=w)
        opt=minimize(loss,start,method='Powell',bounds=[(0,None) if i==1 else (None,None) for i in ids],options={'maxiter':1500,'xtol':1e-7,'ftol':1e-8})
    if not opt.success or not np.isfinite(opt.x).all():raise ValueError('fit failed: '+str(opt.message))
    coef=np.zeros(3);coef[ids]=opt.x
    z=X[:,1:];sd=z.std(axis=0)
    cond=float(np.linalg.cond(np.c_[np.ones(len(z)),(z-z.mean(axis=0))/np.where(sd>0,sd,1)]))
    corr=float(np.corrcoef(z.T)[0,1]) if (sd>0).all() else np.nan
    ident='WEAK_IDENTIFICATION' if cond>30 or not np.isfinite(corr) or abs(corr)>.95 else 'ASSOCIATIONAL_ONLY'
    return dict(coef=coef.tolist(),ref=ref,form=form,tau=tau,n=len(d),families=d.family_id.nunique(),
        quarters=pd.to_datetime(d.date).dt.to_period('Q').nunique(),condition_number=cond,logc_time_correlation=corr,
        identification_status=ident,C_support=[float(d.compute_flops.min()),float(d.compute_flops.max())],
        date_support=[str(d.date.min()),str(d.date.max())],score_support=[float(y.min()),float(y.max())])

def predict(model,C,date):
    a,b,d=model['coef'];t=(pd.to_datetime(date)-pd.Timestamp(model['ref']['date'])).total_seconds()/(365.25*86400)
    return 100*expit(a+b*(np.log(C)-model['ref']['logc'])+d*t)

def gate(d,cfg,frontier=False):
    return len(d)>=(cfg['min_frontier_n'] if frontier else cfg['min_model_n']) and d.family_id.nunique()>=cfg['min_families'] and pd.to_datetime(d.date).dt.to_period('Q').nunique()>=cfg['min_quarters']

def path(d,start=None,end=None):
    d=d.sort_values(['date','model_id']);rows=[]
    for end in pd.period_range(start if start is not None else pd.to_datetime(d.date).min(),end if end is not None else pd.to_datetime(d.date).max(),freq='Q'):
        part=d[pd.to_datetime(d.date)<=end.end_time]
        if part.empty:raise ValueError('bootstrap path lacks a model at fixed window start')
        best=part.sort_values(['average_official','date','model_id'],ascending=[False,True,True]).iloc[0]
        rows.append(dict(date=end.end_time.normalize(),model_id=best.model_id,compute_flops=best.compute_flops,average_official=best.average_official))
    return pd.DataFrame(rows)

def segment(model,C0,C1,t0,t1):
    a,b,d=model['coef'];h=(pd.Timestamp(t1)-pd.Timestamp(t0)).total_seconds()/(365.25*86400)
    if h<=0:raise ValueError('path times must increase')
    dx=np.log(C1/C0);f0=float(predict(model,C0,t0));f1=float(predict(model,C1,t1));v=b*dx+d*h
    if abs(v)>1e-8:ic=b*dx/v*(f1-f0);it=d*h/v*(f1-f0)
    else:
        # Stable integration also covers exact cancellation with nonzero opposing effects.
        base=logit(np.clip(f0/100,1e-15,1-1e-15))
        J=quad(lambda u:100*expit(base+v*u)*(1-expit(base+v*u)),0,1,epsabs=1e-11)[0]
        ic=b*dx*J;it=d*h*J
    z0=a+b*(np.log(C0)-model['ref']['logc'])+d*(pd.Timestamp(t0)-pd.Timestamp(model['ref']['date'])).total_seconds()/(365.25*86400)
    numeric=quad(lambda u:(b*dx+d*h)*100*expit(z0+v*u)*(1-expit(z0+v*u)),0,1,epsabs=1e-10)[0]
    f10=float(predict(model,C1,t0));f01=float(predict(model,C0,t1))
    return dict(model_delta=f1-f0,scale=ic,time=it,integration_error=ic+it-(f1-f0),numerical_check_error=numeric-(f1-f0),
        scale_first=f10-f0,scale_last=f1-f01,shapley_scale=.5*(f10-f0+f1-f01),interaction=f1-f10-f01+f0)

def decompose(model,p):
    if len(p)<2:raise ValueError('fewer than two quarterly path nodes')
    rows=[]
    for j in range(1,len(p)):
        r0=p.iloc[j-1];r1=p.iloc[j];v=segment(model,r0.compute_flops,r1.compute_flops,r0.date,r1.date)
        v.update(start=r0.date,end=r1.date,model_start=r0.model_id,model_end=r1.model_id,
            observed_delta=r1.average_official-r0.average_official)
        v['residual']=v['observed_delta']-v['model_delta'];rows.append(v)
    r=pd.DataFrame(rows)
    if (r.integration_error.abs()>1e-6*np.maximum(1,r.model_delta.abs())).any():raise ArithmeticError('path integral closure failed')
    return r

def validation(d,cfg):
    rows=[];splits=[]
    for fam in d.family_id.unique():splits.append(('family',str(fam),d[d.family_id!=fam],d[d.family_id==fam]))
    quarters=sorted(pd.to_datetime(d.date).dt.to_period('Q').unique())
    for q in quarters[2:]:
        cut=q.start_time;train=d[d.date<cut];test=d[(d.date>=cut)&(d.date<=q.end_time)]
        splits.append(('time_reconstructed',str(q),train,test))
    for kind,name,tr,te in splits:
        if len(tr)<10 or len(te)==0 or tr.family_id.nunique()<3:continue
        for form in FORMS:
            try:
                model=fit(tr,form);pred=np.array([predict(model,r.compute_flops,r.date) for r in te.itertuples()]);err=pred-te.average_official.to_numpy()
                rows.append(dict(validation=kind,fold=name,form=form,n_train=len(tr),n_test=len(te),MAE=np.abs(err).mean(),RMSE=np.sqrt(np.mean(err**2)),status='RECONSTRUCTED_RELEASE_TIME_NOT_REALTIME'))
            except (ValueError,ArithmeticError) as exc:rows.append(dict(validation=kind,fold=name,form=form,status='FAILED',reason=str(exc)))
    return rows

def run(d,out,cfg,rng):
    models={};decomps=[];bootrows=[];cv=[];paths=[]
    for typ,g in d.groupby('model_type'):
        info=dict(n=len(g),families=g.family_id.nunique(),quarters=pd.to_datetime(g.date).dt.to_period('Q').nunique())
        if not gate(g,cfg):models[typ]=dict(status='INSUFFICIENT_IDENTIFICATION_SAMPLE',**info);continue
        try:model=fit(g)
        except ValueError as exc:models[typ]=dict(status='FIT_FAILED',reason=str(exc),**info);continue
        model['status']='FITTED';model['coefficient_meaning']='d is non-scale time association, not an isolated causal technology effect';models[typ]=model
        vrows=validation(g,cfg)
        cv.extend([dict(model_type=typ,**r) for r in vrows])
        vv=pd.DataFrame(vrows)
        if len(vv) and 'MAE' in vv:
            scores=vv.groupby('form').MAE.mean().dropna().to_dict()
            if scores:
                best=min(scores,key=scores.get)
                selected=next(f for f in ['intercept','scale','time','joint'] if f in scores and scores[f]<=scores[best]+.1)
                model['validation_selected_form']=selected
                model['joint_model_supported_by_selection']=selected=='joint'
                model['validation_mean_fold_MAE']=scores
        p=path(g);p['model_type']=typ;paths.append(p)
        if len(p)<2:continue
        dec=decompose(model,p);tot=dec.select_dtypes(include=np.number).sum().to_dict();valid=[]
        for b in range(cfg['bootstrap']):
            try:
                bs=resample_clusters(g,rng);bm=fit(bs,ref=model['ref']);bp=path(bs,start=p.date.min(),end=p.date.max())
                bdec=decompose(bm,bp).select_dtypes(include=np.number).sum()
                fixed=decompose(bm,p).select_dtypes(include=np.number).sum()
                row=dict(model_type=typ,replicate=b,status='OK',a=bm['coef'][0],b=bm['coef'][1],d=bm['coef'][2],**bdec.to_dict())
                row.update({f'fixed_{k}':fixed[k] for k in ['scale','time','model_delta']});valid.append(row);bootrows.append(row)
            except (ValueError,ArithmeticError) as exc:bootrows.append(dict(model_type=typ,replicate=b,status='FAILED',reason=str(exc)))
        B=pd.DataFrame(valid);tot.update(model_type=typ,path='quarterly_logC_linear',start=p.date.min(),end=p.date.max(),bootstrap_valid=len(B),bootstrap_requested=cfg['bootstrap'])
        stable=model.get('joint_model_supported_by_selection',False) and len(B)>=max(20,.8*cfg['bootstrap']) and abs(tot['model_delta'])>=.1 and model['identification_status']!='WEAK_IDENTIFICATION'
        if len(B):
            for k in ['scale','time','model_delta','fixed_scale','fixed_time','fixed_model_delta']:
                for q,label in [(.05,'p05'),(.5,'p50'),(.95,'p95')]:tot[f'{k}_{label}']=B[k].quantile(q)
            stable=stable and (B.model_delta.quantile(.05)>0 or B.model_delta.quantile(.95)<0)
        tot['share_status']='ASSOCIATIONAL_PATH_SHARES' if stable else 'UNSTABLE_OR_WEAK_IDENTIFICATION'
        for k in ['scale','time']:
            tot[k+'_share']=tot[k]/tot['model_delta'] if stable else None
            if stable:
                ratios=B[k]/B.model_delta
                tot[k+'_share_p05']=ratios.quantile(.05);tot[k+'_share_p95']=ratios.quantile(.95)
        decomps.append(tot);dec['model_type']=typ;csv(out/f'tables/path_segments_{typ}.csv',dec)
    dump(out/'interface/P4_decomp_models.json',models)
    csv(out/'interface/P4_decomposition.csv',decomps,columns=None if decomps else ['model_type','status','scale','time','model_delta'])
    csv(out/'tables/decomposition_bootstrap.csv',bootrows,columns=None if bootrows else ['model_type','status'])
    csv(out/'tables/model_validation.csv',cv,columns=None if cv else ['model_type','status'])
    csv(out/'tables/champion_paths.csv',pd.concat(paths,ignore_index=True) if paths else pd.DataFrame(columns=['model_type','date','model_id']))
    return models

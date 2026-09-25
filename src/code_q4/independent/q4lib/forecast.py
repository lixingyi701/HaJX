import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from .common import *
from .dynamics import fit,predict,gate
from .audit import eligible_at

def observed(m,origin):
    rows=[]
    for scope in ['strict_full','strict_compute','broad_full']:
        d=eligible_at(m,origin,panel=scope=='strict_compute',broad=scope=='broad_full')
        for typ,g in d.groupby('model_type'):
            for period in pd.period_range(g.date.min(),origin,freq='Q'):
                cut=min(period.end_time.normalize(),origin);past=g[g.date<=cut]
                if past.empty:continue
                best=past.sort_values(['average_official','date','model_id'],ascending=[False,True,True]).iloc[0]
                recent=past[past.date>=cut-pd.DateOffset(months=12)].average_official.nlargest(5)
                rows.append(dict(scope=scope,model_type=typ,date=cut,frontier=best.average_official,champion=best.model_id,
                    top5_12m=recent.mean(),top5_n=len(recent),date_interpretation='reconstructed_release_or_submission'))
    return pd.DataFrame(rows)

def compute_growth(e,origin,cfg):
    d=e.copy();d['date']=dates(d['Publication date']);d['C']=pd.to_numeric(d['Training compute (FLOP)'],errors='coerce')
    domain=d.Domain.fillna('').str.lower().str.contains('language')
    d=d[domain & (d['Open model weights?'].astype(str).str.lower()=='yes') & (d.C>0) & (d.date<=origin)]
    # C4 lacks uniform license evidence; this is explicitly a broad compute proxy.
    d['quarter']=d.date.dt.to_period('Q');rows=[]
    for q,g in d.groupby('quarter'):
        rows.append(dict(quarter=str(q),date=min(q.end_time.normalize(),origin),n=len(g),logc_q90=np.log(g.C).quantile(.9),logc_max=np.log(g.C).max()))
    q=pd.DataFrame(rows)
    enough=q[q.n>=10].copy() if len(q) else q
    result={'scope':'C4_open_weights_language_broad_proxy_license_unverified','status':'INSUFFICIENT_QUARTERS','origin':str(origin),
        'growth':cfg.get('assumed_growth'),'growth_source':'ASSUMED_SCENARIO' if cfg.get('assumed_growth') is not None else 'UNAVAILABLE',
        'asof_status':'RECONSTRUCTED_C4_SNAPSHOT_NOT_HISTORICAL_ARCHIVE'}
    if len(enough)>=4:
        t=years(enough.date,enough.date.min()).to_numpy();fitg=theilslopes(enough.logc_q90,t,.90)
        result.update(status='DATA_ESTIMATE_BROAD_PROXY',growth=float(fitg.slope),growth_p05=float(fitg.low_slope),growth_p95=float(fitg.high_slope),growth_source='DATA_ESTIMATE')
        result['slowdown_label_valid']=bool(fitg.slope>0 and fitg.low_slope>0)
    return result,q

def run(m,e,models,out,cfg,rng,origin):
    obs=observed(m,origin);csv(out/'tables/observed_frontiers.csv',obs)
    growth,cq=compute_growth(e,origin,cfg);csv(out/'tables/C4_compute_quarters.csv',cq);dump(out/'tables/compute_growth.json',growth)
    rows=[];fmodels={};failures=[];bt=[]
    panel=eligible_at(m,origin,panel=True)
    for typ in ['base','chat']:
        g=panel[panel.model_type==typ]
        if not gate(g,cfg):
            rows.append(dict(model_type=typ,status='INSUFFICIENT_MODEL_SAMPLE',origin=origin,n=len(g),families=g.family_id.nunique()));continue
        taus=[.75,.9,.95] if gate(g,cfg,True) else [None]
        x0=float(np.log(g.compute_flops).quantile(.9));champ=float(eligible_at(m,origin).query('model_type == @typ').average_official.max())
        for tau in taus:
            selected_form=models.get(typ,{}).get('validation_selected_form','joint') if tau is None else 'joint'
            fm=fit(g,form=selected_form,tau=tau);fm['selection_reason']='validation-selected simpler mean model' if tau is None else 'predeclared quantile response'
            key=typ+'_'+str(tau);fmodels[key]=fm
            target='CONDITIONAL_QUANTILE' if tau else 'CONDITIONAL_MEAN_NOT_FRONTIER'
            if growth['growth'] is None:
                rows.append(dict(model_type=typ,status='NO_COMPUTE_GROWTH_ASSUMPTION',origin=origin,target_definition=target));continue
            bs=[]
            for b in range(cfg['bootstrap']):
                try:
                    sample=resample_clusters(g,rng);bm=fit(sample,form=selected_form,tau=tau,ref=fm['ref']);bx=float(np.log(sample.compute_flops).quantile(.9));bs.append((bm,bx))
                except ValueError as exc:failures.append(dict(model_type=typ,tau=tau,replicate=b,reason=str(exc)))
            future=[origin+pd.DateOffset(months=k) for k in range(25)]
            for cm in [1.,.5,0.]:
                for tm in [1.,.5,0.]:
                    gc=growth['growth']*cm;central=[];draws=[]
                    for date in future:
                        h=(date-origin).days/365.25;C=np.exp(x0+gc*h)
                        ff=dict(fm);ff['coef']=fm['coef'].copy()
                        # Retain fitted historical level and change only the future time increment.
                        zdate=origin+pd.Timedelta(days=(date-origin).days*tm)
                        central.append(float(predict(ff,C,zdate)))
                    for bm,bx in bs:
                        vals=[]
                        for date in future:
                            h=(date-origin).days/365.25;zdate=origin+pd.Timedelta(days=(date-origin).days*tm)
                            vals.append(float(predict(bm,np.exp(bx+gc*h),zdate)))
                        draws.append(vals)
                    qq=np.quantile(draws,[.05,.25,.5,.75,.95],axis=0) if draws else np.full((5,25),np.nan)
                    cumulative=np.maximum.accumulate(np.maximum(central,champ))
                    for k,date in enumerate(future):
                        h=(date-origin).days/365.25
                        rows.append(dict(chain='C_empirical',model_type=typ,status='CONDITIONAL_SCENARIO',origin=origin,target=date,horizon_months=k,tau=tau,
                            target_definition=target,response_form=selected_form,anchor_mode='raw_conditional',compute_multiplier=cm,technology_multiplier=tm,
                            log_compute_growth_per_year=gc,compute_flops=np.exp(x0+gc*h),central=central[k],p05=qq[0,k],p25=qq[1,k],p50=qq[2,k],p75=qq[3,k],p95=qq[4,k],
                            observed_anchor=champ,anchor_bias=central[0]-champ,cumulative_conditional_attainable=cumulative[k],
                            bootstrap_valid=len(bs),uncertainty_scope='cluster_parameter_and_panel_compute_anchor_only; growth conditional; no maximum_distribution',
                            extrapolates_compute=bool(np.exp(x0+gc*h)>fm['C_support'][1]),extrapolates_time=True))
        # Honest horizon tests; all model fits use evaluations available by each historical origin.
        for period in pd.period_range(m.eval_date.min(),origin,freq='2Q'):
            bo=period.end_time.normalize()
            if bo>=origin:continue
            tr=eligible_at(m,bo,panel=True);tr=tr[tr.model_type==typ]
            for months in [12,24]:
                end=bo+pd.DateOffset(months=months)
                if end>m.eval_date.max():bt.append(dict(model_type=typ,origin=bo,horizon_months=months,status='INSUFFICIENT_HORIZON'));continue
                if not gate(tr,cfg):bt.append(dict(model_type=typ,origin=bo,horizon_months=months,status='INSUFFICIENT_TRAINING'));continue
                future_eligible=eligible_at(m,end);future_eligible=future_eligible[future_eligible.model_type==typ]
                start_eligible=eligible_at(m,bo);start_eligible=start_eligible[start_eligible.model_type==typ]
                if future_eligible.empty or start_eligible.empty:continue
                actual=future_eligible.average_official.max();anchor=start_eligible.average_official.max()
                bt.append(dict(model_type=typ,origin=bo,horizon_months=months,method='flat',actual=actual,prediction=anchor,error=anchor-actual,status='RECONSTRUCTED_NOT_REALTIME'))
    if not bt:
        bt=[dict(model_type=t,horizon_months=h,status='INSUFFICIENT_EVALUATION_HISTORY',reason='C2 evaluation snapshot spans less than 12 months') for t in ['base','chat'] for h in [12,24]]
    csv(out/'interface/P4_frontier_forecast.csv',rows);dump(out/'interface/P4_frontier_models.json',fmodels)
    csv(out/'tables/forecast_bootstrap_failures.csv',failures,columns=['model_type','tau','replicate','reason']);csv(out/'tables/frontier_backtest.csv',bt)
    return rows

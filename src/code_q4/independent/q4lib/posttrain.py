import numpy as np
import pandas as pd
from .common import *

def run(m,l,out,cfg,rng,origin):
    eligible=m[(m.eval_date<=origin)&(m.date<=origin)&m.include_primary].set_index('model_id')
    # A name suffix is only a candidate. Confirm parent and evaluation equivalence in overrides.
    candidates=[]
    for mid,r in eligible[eligible.model_type=='chat'].iterrows():
        for suffix in ['-Instruct','-Chat','-it','-instruct','-chat']:
            base=mid.removesuffix(suffix)
            if base!=mid and base in eligible.index:candidates.append(dict(base_id=base,chat_id=mid,status='NAME_CANDIDATE_NOT_VERIFIED',evidence=''))
    csv(out/'tables/pair_candidates.csv',candidates,columns=['base_id','chat_id','status','evidence'])
    effects=[];fail=[]
    p=pd.read_csv(cfg['pairs']).fillna('') if cfg.get('pairs') else pd.DataFrame()
    for _,r in p.iterrows():
        if not r.get('evidence') or str(r.get('verified','')).lower() not in ['true','1','yes']:
            fail.append(dict(base_id=r.base_id,chat_id=r.chat_id,reason='UNVERIFIED_PARENT'));continue
        if r.base_id not in eligible.index or r.chat_id not in eligible.index:
            fail.append(dict(base_id=r.base_id,chat_id=r.chat_id,reason='NOT_ELIGIBLE_AT_ORIGIN'));continue
        b=eligible.loc[r.base_id];c=eligible.loc[r.chat_id]
        if b.model_type!='base' or c.model_type!='chat' or str(r.get('same_eval_settings','')).lower() not in ['true','1','yes']:
            fail.append(dict(base_id=r.base_id,chat_id=r.chat_id,reason='TYPE_OR_EVAL_SETTINGS_NOT_CONFIRMED'));continue
        for dim in DIMS:effects.append(dict(base_id=r.base_id,chat_id=r.chat_id,dimension=dim,base_score=b[dim],chat_score=c[dim],difference=c[dim]-b[dim],evidence=r.evidence))
    ef=pd.DataFrame(effects,columns=['base_id','chat_id','dimension','base_score','chat_score','difference','evidence'])
    csv(out/'interface/P4_posttrain_effects.csv',ef);csv(out/'tables/pair_rejections.csv',fail,columns=['base_id','chat_id','reason'])
    summary=[]
    for dim,g in ef.groupby('dimension'):
        parent=g.groupby('base_id').difference.mean().to_numpy();means=[rng.choice(parent,len(parent),replace=True).mean() for _ in range(cfg['bootstrap'])]
        summary.append(dict(dimension=dim,n_pairs=len(g),independent_parents=len(parent),mean=parent.mean(),median=np.median(parent),
            p05=np.quantile(means,.05) if len(parent)>=10 and means else None,p95=np.quantile(means,.95) if len(parent)>=10 and means else None,
            status='PARENT_CLUSTER_INTERVAL_NO_SIGNIFICANCE_CLAIM' if len(parent)>=10 else 'SMALL_PAIRED_SAMPLE'))
    dump(out/'tables/posttrain_status.json',{'status':'VERIFIED_PAIRS_ANALYZED' if len(ef) else 'NO_VERIFIED_PAIRS','candidate_pairs':len(candidates),'summary':summary,
        'interpretation':'Associations of post-training; no preassigned direction, no isolated alignment-algorithm causal claim'})

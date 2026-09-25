import json,re
import numpy as np
import pandas as pd
from .common import *

FAMILIES={'ifeval':'IFEval','bbh':'BBH','math':'MATH Lvl 5','gpqa':'GPQA','musr':'MUSR','mmlu_pro':'MMLU-PRO'}
EXPECTED={'ifeval':1,'bbh':24,'math':7,'gpqa':3,'musr':3,'mmlu_pro':1}
MUSR={'murder_mysteries':2,'object_placements':5,'team_allocation':3}

def parse_document(d,path):
    mid=d.get('model_name')
    if not mid:
        args=d.get('config',{}).get('model_args','')
        match=re.search(r'pretrained=([^,]+)',str(args));mid=match.group(1) if match else None
    if not mid:raise ValueError('model identity missing')
    t=pd.to_datetime(d.get('date'),unit='s',errors='coerce')
    if pd.isna(t):raise ValueError('evaluation time missing')
    config=d.get('config',{});revision=config.get('model_sha') or config.get('model_revision') or 'UNKNOWN'
    rows=[];rules=[]
    groups=d.get('group_subtasks',{})
    for task,r in d.get('results',{}).items():
        # Groups are never counted as independent observations.
        if groups.get(task):continue
        short=task.removeprefix('leaderboard_')
        fam=next((f for f in FAMILIES if short==f or short.startswith(f+'_')),None)
        if fam is None:continue
        cfg=d.get('configs',{}).get(task,{})
        metrics=['prompt_level_strict_acc,none','inst_level_strict_acc,none'] if fam=='ifeval' else [
            'exact_match,none' if fam=='math' else ('acc,none' if fam=='mmlu_pro' else 'acc_norm,none')]
        # Some MATH runs use the filtered flexible extraction metric.
        if fam=='math' and metrics[0] not in r:
            metrics=[k for k in r if k.startswith('exact_match,') and 'stderr' not in k]
            if len(metrics)!=1:raise ValueError('ambiguous MATH metric: '+task)
        if not all(k in r for k in metrics):continue
        score=float(np.mean([r[k] for k in metrics]))
        choice=cfg.get('doc_to_choice');chance=None;source=''
        if fam=='bbh' and isinstance(choice,list) and len(choice)>1:chance=1/len(choice);source='C8 configs.doc_to_choice'
        elif fam=='gpqa':chance=.25;source='v2 GPQA four-option task convention; inspect version'
        elif fam=='mmlu_pro':chance=.1;source='v2 MMLU-Pro ten-option task convention'
        elif fam=='musr':
            n=next((v for k,v in MUSR.items() if short.endswith(k)),None)
            if n:chance=1/n;source='v2 MuSR task-specific choice convention'
        elif fam in ['ifeval','math']:chance=0.;source='score floor convention; not random-generation probability'
        n=d.get('n-samples',{}).get(task,{}).get('effective',np.nan)
        raw_norm=(score-chance)/(1-chance) if chance is not None else np.nan
        parent_key='leaderboard_math_hard' if fam=='math' else 'leaderboard_'+fam
        parent=d.get('results',{}).get(parent_key,{})
        parent_rate=float(np.mean([parent[k] for k in metrics])) if all(k in parent for k in metrics) else np.nan
        rows.append(dict(model_id=mid,revision=revision,eval_date=t,run_file=str(path),task=task,family=fam,
            version=d.get('versions',{}).get(task),task_hash=d.get('task_hashes',{}).get(task),
            metric=' + '.join(metrics),parent_raw_rate=parent_rate,raw_rate=score,chance=chance,normalized_unclipped=raw_norm,
            normalized_clipped=max(0,raw_norm) if np.isfinite(raw_norm) else np.nan,n_samples=n))
        rules.append(dict(task=task,version=d.get('versions',{}).get(task),metric=metrics,chance=chance,
            choice_rule=source,normalization='(raw-c)/(1-c)',clipping_rule='both retained; clipped candidate requires C2 verification',
            aggregation_weight='n_samples' if fam in ['math','gpqa'] else 'equal_tasks',
            warning='GPQA subsets overlap; effective counts are aggregation weights, not independent sample counts' if fam=='gpqa' else ''))
    if not rows:raise ValueError('no supported leaf tasks')
    return rows,rules

def parse_all(data,out):
    rows=[];errors=[];rules={};runs=[]
    paths=sorted(data.rglob('*.json'))
    for p in paths:
        try:
            d=json.loads(p.read_text(encoding='utf-8'));rr,ru=parse_document(d,p.relative_to(data))
            rows.extend(rr);runs.append({'model_id':rr[0]['model_id'],'run_file':str(p.relative_to(data)),'n_tasks':len(rr),'eval_date':rr[0]['eval_date']})
            for r in ru:rules[(r['task'],str(r['version']),str(r['metric']),str(r['chance']))]=r
        except (ValueError,KeyError,TypeError,OverflowError) as exc:errors.append({'file':str(p.relative_to(data)),'reason':str(exc)})
    l=pd.DataFrame(rows)
    if l.empty:raise ValueError('No usable C8 JSON evaluations')
    csv(out/'tables/C8_runs.csv',runs);csv(out/'tables/C8_parse_failures.csv',errors,columns=['file','reason'])
    dump(out/'interface/P4_score_rules.json',{'status':'CANDIDATE_RULES_WITH_EXPLICIT_REPLICATION_AUDIT','rules':list(rules.values()),'dimension_weights':{d:1/6 for d in DIMS}})
    return l

def score_audit(l,m,out,origin,cfg):
    # One intact evaluation per model; do not splice tasks across revisions/runs.
    runs=l[l.eval_date<=origin].groupby(['model_id','run_file']).agg(eval_date=('eval_date','first'),n_tasks=('task','nunique')).reset_index()
    chosen=runs.sort_values(['eval_date','run_file']).drop_duplicates('model_id',keep='last')
    q=l.merge(chosen[['model_id','run_file']],on=['model_id','run_file'])
    csv(out/'tables/C8_leaf_scores.csv',q)
    rows=[]
    for (mid,fam),d in q.groupby(['model_id','family']):
        complete=d.task.nunique()==EXPECTED[fam] and d.normalized_clipped.notna().all()
        w=d.n_samples.to_numpy(float) if fam in ['math','gpqa'] else np.ones(len(d))
        complete=complete and np.isfinite(w).all() and (w>0).all()
        row=dict(model_id=mid,dimension=FAMILIES[fam],n_tasks=len(d),complete=complete)
        for col in ['raw_rate','normalized_unclipped','normalized_clipped']:
            row[col+'_aggregate']=100*np.average(d[col],weights=w) if complete else np.nan
        row['reported_parent_raw']=100*d.parent_raw_rate.iloc[0]
        row['leaf_parent_raw_difference']=row['raw_rate_aggregate']-row['reported_parent_raw']
        rows.append(row)
    agg=pd.DataFrame(rows); official=m[['model_id','eval_date']+DIMS].melt(id_vars=['model_id','eval_date'],var_name='dimension',value_name='official')
    chk=agg.merge(official,on=['model_id','dimension']);chk['difference']=chk.normalized_clipped_aggregate-chk.official
    chk['verified']=chk.complete & (chk.difference.abs()<=cfg['score_tolerance'])
    chk['comparison_status']=np.where(chk.verified,'REPRODUCED_WITHIN_TOLERANCE','MISMATCH_OR_VERSION_UNKNOWN')
    csv(out/'tables/C8_C2_replication.csv',chk)
    csv(out/'tables/C8_replication_summary.csv',chk.groupby('dimension').agg(n=('model_id','size'),n_verified=('verified','sum'),MAE=('difference',lambda s:s.abs().mean()),max_error=('difference',lambda s:s.abs().max())).reset_index())
    sat=[]
    for (task,version),d in q.groupby(['task','version'],dropna=False):
        vals=d.normalized_clipped.dropna()
        sat.append(dict(task=task,version=version,origin=origin,n_models=d.model_id.nunique(),
            median=vals.median(),top5_mean=vals.nlargest(5).mean(),n_top=min(5,len(vals)),
            threshold=.85,saturated=bool(len(vals)>=5 and vals.nlargest(5).mean()>=.85),
            status='DESCRIPTIVE_CANDIDATE_SCALE_NOT_C2_REPLACEMENT'))
    csv(out/'interface/P4_task_saturation.csv',sat)
    sens=[]
    mm=m[(m.eval_date<=origin)&(m.date<=origin)&m.include_primary]
    for typ,g in mm.groupby('model_type'):
        for dim in DIMS:
            score='double_weight_'+dim
            best=g.sort_values([score,'model_id'],ascending=[False,True]).iloc[0]
            sens.append(dict(model_type=typ,weighted_dimension=dim,n=len(g),spearman_with_official=g[score].corr(g.average_official,method='spearman'),
                champion=best.model_id,weighted_score=best[score],official_score=best.average_official))
    csv(out/'tables/score_weight_sensitivity.csv',sens,columns=None if sens else ['model_type','weighted_dimension','n'])
    return chk,q

import re
import numpy as np
import pandas as pd
from .common import *

def build_index(epoch):
    exact, candidate={},{}
    for i,r in epoch.iterrows():
        ids=re.findall(r'https?://huggingface.co/([\w.-]+/[\w.-]+)',str(r.get('Link',''))+' '+str(r.get('Reference','')))
        if '/' in str(r.Model) and ' ' not in str(r.Model):ids.append(str(r.Model))
        for key in ids:exact.setdefault(key.lower().rstrip('.,'),set()).add(i)
        candidate.setdefault(norm(str(r.Model).split('/')[-1]),set()).add(i)
    return exact,candidate

def match(model,epoch,index):
    exact,cand=index;ids=exact.get(model.lower(),set())
    if len(ids)==1:return next(iter(ids)),'EXACT_HF_LINK'
    if len(ids)>1:return None,'AMBIGUOUS_EXACT'
    # Issuer identity must be documented; a same basename in another account is not an identity match.
    candidates=cand.get(norm(model.split('/')[-1]),set())
    owner=model.split('/')[0].lower()
    ok=[i for i in candidates if owner in re.findall(r'[\w.-]+',str(epoch.loc[i].get('Hugging Face developer id','')).lower())]
    if len(ok)==1:return ok[0],'EXACT_NAME_AND_DEVELOPER'
    return None,'UNVERIFIED_NAME_CANDIDATE' if candidates else 'UNMATCHED'

def build(data,out,cfg,c8models):
    a=pd.read_csv(data/'leaderboard_enhanced.csv');e=pd.read_csv(data/'epoch_all_ai_models.csv')
    for col in DIMS+[AVG]:a[col]=pd.to_numeric(a[col],errors='coerce')
    a['eval_date']=dates(a['Submission Date'])
    maxdate=a.eval_date.max(); origin=pd.Timestamp(cfg['origin']) if cfg['origin'] else (maxdate.to_period('Q')-1).end_time.normalize()
    cutoff=pd.Timestamp(cfg['cutoff']) if cfg['cutoff'] else maxdate.normalize()
    if origin>cutoff:raise ValueError('forecast origin must not exceed data cutoff')
    a['model_id']=a.Model.astype(str);a['model_type']=a.Type.map(model_type)
    # Conflicting base/chat labels for the same unknown revision cannot be disambiguated.
    conflict=a.groupby('model_id').model_type.nunique();conflict=set(conflict[conflict>1].index)
    csv(out/'tables/duplicate_rows.csv',a[a.duplicated('model_id',keep=False)])
    a=a[a.eval_date.notna() & (a.eval_date<=cutoff)].sort_values('eval_date').drop_duplicates('model_id',keep='last').copy()
    ix=build_index(e);records=[]
    overrides=pd.read_csv(cfg['overrides']).fillna('') if cfg.get('overrides') else pd.DataFrame()
    if len(overrides) and overrides.model_id.duplicated().any():raise ValueError('duplicate override model_id')
    ov=overrides.set_index('model_id').to_dict('index') if len(overrides) else {}
    for _,r in a.iterrows():
        mid=r.model_id;i,grade=match(mid,e,ix); z=e.loc[i] if i is not None else pd.Series(dtype=object)
        manual=ov.get(mid,{})
        if manual and not manual.get('evidence'):raise ValueError(f'override lacks evidence: {mid}')
        if manual.get('epoch_model'):
            found=e.index[e.Model==manual['epoch_model']]
            if len(found)!=1:raise ValueError(f'override epoch model not unique: {mid}')
            z=e.loc[found[0]];grade='VERIFIED_ALIAS'
        compute=pd.to_numeric(z.get('Training compute (FLOP)'),errors='coerce')
        pub=pd.to_datetime(z.get('Publication date'),errors='coerce')
        # C2 enriched dates from older approximate matching are only candidates, not authority.
        rd=pub if pd.notna(pub) else pd.NaT
        lic=str(r['Hub License']).lower();lic_ok=lic in cfg['license_allowlist']
        if manual.get('license_approved')!='':
            if str(manual.get('license_approved','')).lower() in ('true','1','yes'):lic_ok=True
        open_c4=str(z.get('Open model weights?','')).lower()
        in_c8=mid in c8models
        evidence_date=c8models.get(mid,pd.NaT) if isinstance(c8models,dict) else pd.NaT
        weights='YES_C4' if open_c4=='yes' else ('YES_C8_EVALUATED' if in_c8 else 'UNKNOWN')
        if open_c4=='no':weights='CONFLICT' if in_c8 else 'NO'
        if str(manual.get('weights_approved','')).lower() in ('true','1','yes'):weights='YES_VERIFIED'
        mtype=manual.get('model_type') or r.model_type
        type_conflict=mid in conflict and not manual.get('model_type')
        rebuilt=r[DIMS].mean() if r[DIMS].notna().all() else np.nan
        avg_ok=pd.notna(rebuilt) and abs(rebuilt-r[AVG])<=cfg['score_tolerance']
        valid_scores=r[DIMS].notna().all() and r[DIMS].between(0,100).all() and avg_ok
        reasons=[]
        if not weights.startswith('YES'):reasons.append('WEIGHTS_UNKNOWN_OR_CONFLICT')
        if not lic_ok:reasons.append('LICENSE_UNREVIEWED')
        if mtype not in ['base','chat']:reasons.append('TYPE_EXCLUDED')
        if type_conflict:reasons.append('TYPE_CONFLICT_UNKNOWN_REVISION')
        if not valid_scores:reasons.append('SCORE_INCOMPLETE_OR_MISMATCH')
        row=r.to_dict();row.update(family_id=manual.get('family_id') or family(mid),model_type=mtype,
            revision='UNKNOWN_C2_SNAPSHOT',benchmark_version='C2_snapshot_v2_unverified_revision',
            release_date=rd,date=rd if pd.notna(rd) else r.eval_date,date_basis='RELEASE' if pd.notna(rd) else 'SUBMISSION_FALLBACK',
            weights_status=weights,weights_evidence_date=evidence_date if weights=='YES_C8_EVALUATED' else pd.NaT,license_id=lic,research_use_status='ALLOWLIST_METADATA' if lic_ok else 'UNREVIEWED',
            reproduction_status='WEIGHTS_EVALUATED' if in_c8 else 'NOT_VERIFIED',
            license_evidence=manual.get('evidence') or 'C2 Hub License metadata; license text/version not independently archived',
            include_primary=not reasons,include_broad=valid_scores and mtype in ['base','chat'] and not type_conflict and weights not in ['NO','CONFLICT'],
            exclusion_reason=';'.join(reasons),N=r['#Params (B)']*1e9,D=z.get('Training dataset size (total)'),
            compute_flops=compute,compute_source='C4:'+str(z.get('Model','')) if pd.notna(compute) else '',
            compute_scope='C4_TRAINING_SCOPE_NOT_FULLY_VERIFIED',is_compute_estimated=str(z.get('Training compute estimation method','UNKNOWN')),
            match_grade=grade,epoch_model=z.get('Model'),metadata_available_date=z.get('Last modified'),
            metadata_time_status='RECONSTRUCTED_NOT_ASOF_ARCHIVE',base_parent=z.get('Base model'),
            context_train=np.nan,average_official=r[AVG],average_rebuilt=rebuilt,
            min_dimension=r[DIMS].min(),dimension_sd=r[DIMS].std(ddof=0))
        for dim in DIMS:row['double_weight_'+dim]=(r[DIMS].sum()+r[dim])/7 if valid_scores else np.nan
        records.append(row)
    m=pd.DataFrame(records)
    c7=pd.read_csv(data/'model_architecture_metadata.csv').drop_duplicates('model_name')
    m['context_max']=m.model_id.map(c7.set_index('model_name').max_position_embeddings)
    m['include_panel']=m.include_primary & (m.compute_flops>0) & m.release_date.notna()
    csv(out/'interface/P4_model_audit.csv',m)
    csv(out/'tables/filter_flow.csv',[{'stage':k,'n':int(v)} for k,v in {
        'C2_original':len(pd.read_csv(data/'leaderboard_enhanced.csv')),'deduplicated_cutoff':len(m),
        'strict_score_sample':m.include_primary.sum(),'broad_score_sample':m.include_broad.sum(),
        'strict_compute_panel':m.include_panel.sum(),'unknown_release':m.release_date.isna().sum(),
        'unreviewed_license':(m.research_use_status=='UNREVIEWED').sum(),'conflicting_type':len(conflict)}.items()])
    csv(out/'tables/missing_fields.csv',[{'field':c,'missing':int(m[c].isna().sum())} for c in ['compute_flops','release_date','D','context_max','metadata_available_date']])
    # C3 is a separate measurement regime unless cross-calibrated.
    c3=pd.read_csv(data/'leaderboard_extended_timeseries.csv');cols=['IFEval','BBH','MATH_Lvl5','GPQA','MUSR','MMLU_PRO']
    c3['six_mean']=c3[cols].mean(axis=1);c3['average_difference']=c3.Average-c3.six_mean
    c3['use_status']='DESCRIPTIVE_ONLY_MIXED_PROVENANCE';csv(out/'tables/C3_audit.csv',c3)
    c1=pd.read_csv(data/'leaderboard_cleaned.csv').drop_duplicates('Model',keep='last')
    c1check=a[['Model',AVG]].merge(c1[['Model',AVG]],on='Model',suffixes=('_C2','_C1'))
    c1check['difference']=c1check[AVG+'_C2']-c1check[AVG+'_C1'];csv(out/'tables/C1_C2_check.csv',c1check)
    pq=list(data.rglob('*.parquet'));c9={'files':[str(p.relative_to(data)) for p in pq]}
    try:
        c9['tables']=[]
        for p in pq:
            table=pd.read_parquet(p)
            c9['tables'].append({'path':str(p.relative_to(data)),'rows':len(table),'columns':list(table.columns)})
        c9['status']='SCHEMA_CHECK_ONLY_NOT_POOLED'
    except (ImportError,ValueError) as exc:c9.update(status='OPTIONAL_READER_UNAVAILABLE',reason=str(exc))
    dump(out/'tables/C9_audit.json',c9)
    return m,e,origin,cutoff

def eligible_at(m,origin,panel=False,broad=False):
    flag='include_panel' if panel else ('include_broad' if broad else 'include_primary')
    mask=m[flag] & (m.eval_date<=origin) & (m.date<=origin)
    if 'weights_evidence_date' in m:
        mask &= m.weights_evidence_date.isna() | (pd.to_datetime(m.weights_evidence_date)<=origin)
    return m[mask].copy()

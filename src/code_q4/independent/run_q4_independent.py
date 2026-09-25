"""Run: python run_q4_independent.py --data PATH --out NEW_DIRECTORY"""
import argparse,json,sys,platform,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,matplotlib
from q4lib.common import *
from q4lib import audit,scoring,dynamics,bridge,posttrain,forecast,report

def main():
    parser=argparse.ArgumentParser(description='Q4 interpretable independent pipeline; no Q1/Q2/Q3 imports')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='new or empty output directory')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config_q4.json'))
    parser.add_argument('--origin');parser.add_argument('--cutoff');parser.add_argument('--bootstrap',type=int)
    parser.add_argument('--assumed-growth',type=float,help='ln(FLOPs)/year, only if C4 cannot estimate growth')
    parser.add_argument('--overrides',type=Path);parser.add_argument('--pairs',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    for k in ['origin','cutoff','bootstrap','overrides','pairs','assumed_growth']:
        if getattr(args,k,None) is not None:cfg[k]=str(getattr(args,k)) if k in ['overrides','pairs'] else getattr(args,k)
    if cfg['bootstrap']<0:raise ValueError('bootstrap must be nonnegative')
    data=args.data.resolve();out=args.out.resolve()
    if not data.is_dir():raise FileNotFoundError('C data directory not found: '+str(data))
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory is not empty; choose a new --out to preserve previous runs')
    for p in ['interface','tables','figures']:(out/p).mkdir(parents=True,exist_ok=True)
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');rng=np.random.default_rng(cfg['seed'])
    dump(out/'config_frozen.json',cfg)
    print('[1/7] C8 parsing',flush=True);leaves=scoring.parse_all(data,out)
    print('[2/7] identity, eligibility, score and date audits',flush=True)
    m,e,origin,cutoff=audit.build(data,out,cfg,leaves.groupby('model_id').eval_date.min().to_dict())
    # Keep C8 actual timestamp availability distinct from C2 submission timestamp.
    print('[3/7] C8 score replication',flush=True);scoring.score_audit(leaves,m,out,origin,cfg)
    print('[4/7] dynamical fit and exact path attribution',flush=True)
    panel=audit.eligible_at(m,origin,panel=True);models=dynamics.run(panel,out,cfg,rng)
    print('[5/7] loss bridge and verified base/chat pairs',flush=True)
    bridge.run(data,out,cfg,rng,m);posttrain.run(m,leaves,out,cfg,rng,origin)
    print('[6/7] conditional forecasts; Q3 symbolic interface',flush=True)
    forecast.run(m,e,models,out,cfg,rng,origin);report.resource_placeholder(out)
    if cfg.get('run_broad_sensitivity',True):
        print('[6b/7] broad eligibility sensitivity (not primary findings)',flush=True)
        broad=m.copy();broad['include_primary']=broad.include_broad
        broad['include_panel']=broad.include_broad & (broad.compute_flops>0) & broad.release_date.notna()
        bo=out/'sensitivity_broad'
        for sub in ['interface','tables']:(bo/sub).mkdir(parents=True,exist_ok=True)
        bm=dynamics.run(audit.eligible_at(broad,origin,panel=True),bo,cfg,rng)
        forecast.run(broad,e,bm,bo,cfg,rng,origin)
        for fp in bo.rglob('*.csv'):
            frame=pd.read_csv(fp)
            frame['sample_scope']='BROAD_SENSITIVITY_LICENSE_OR_WEIGHTS_UNREVIEWED'
            csv(fp,frame)
        dump(bo/'SCOPE.json',{'status':'EXPLORATORY_SENSITIVITY_ONLY','rule':'C2 broad eligible models with reliable C4 identity match; license/weights may be unreviewed',
            'warning':'These estimates do not replace primary screened results. Conditional means are not frontier predictions.'})
    print('[7/7] figures, report, hashes',flush=True)
    report.figures(out);report.write_report(out,m,origin,cutoff,cfg)
    inputs={str(p.relative_to(data)):sha(p) for p in sorted(data.rglob('*')) if p.is_file()}
    for k in ['overrides','pairs']:
        if cfg.get(k):inputs[k+':'+str(cfg[k])]=sha(cfg[k])
    code_root=Path(__file__).parent
    manifest={'schema_version':SCHEMA,'run_id':run_id,'input_sha256':inputs,
        'code_sha256':{str(p.relative_to(code_root)):sha(p) for p in code_root.rglob('*.py')},
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'config':cfg,'forecast_origin':origin,'data_cutoff':cutoff,
        'targets':[origin+pd.DateOffset(months=h) for h in [12,24]],
        'time_status':'C2 submission plus C4 release reconstructed snapshot; C2 exact eval time/revision unavailable',
        'Q3_status':'DEFERRED_Q3','output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    dump(out/'interface/P4_run_manifest.json',manifest)
    print('Finished:',out,flush=True)

if __name__=='__main__':main()

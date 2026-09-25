from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd

DIMS = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
AVG = 'Average ⬆️'
SCHEMA = 'q4-independent-1.0'

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, Path)): return str(x)
    if x is pd.NaT or x is pd.NA: return None
    return x

def dump(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csv(path, rows, columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    d.to_csv(path,index=False,encoding='utf-8-sig')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def norm(s):
    return re.sub(r'[^a-z0-9.]','',str(s).lower())

def family(s):
    # Conservative issuer clustering avoids pretending sibling generations are independent.
    return str(s).split('/')[0].lower()

def dates(s):return pd.to_datetime(s,errors='coerce',utc=True).dt.tz_convert(None)

def years(s,ref):return (pd.to_datetime(s)-pd.Timestamp(ref)).dt.total_seconds()/(365.25*86400)

def model_type(s):
    s=str(s).lower()
    if 'pretrained' in s:return 'base'
    if 'chat' in s or 'fine-tuned' in s:return 'chat'
    return 'other'

def resample_clusters(d,rng):
    groups=d.family_id.unique()
    picks=rng.choice(groups,len(groups),replace=True)
    return pd.concat([d[d.family_id==g].assign(family_id=f'bootstrap_{i}') for i,g in enumerate(picks)],ignore_index=True)

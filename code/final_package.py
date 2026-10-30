"""Portable integrity contract for the compact final research package.

Training recovery files are intentionally absent. Original manifests remain
provenance; this module checks the separately sealed retained file inventory.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json
import math

DATASETS=('sml','pm25','nasdaq')
SEEDS=(11,23,47,71,101)
KINDS=('enet','rf','xgt','strx','arimax')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def checked_path(root,name):
    name=str(name).replace('\\','/')
    posix=PurePosixPath(name);windows=PureWindowsPath(name)
    if not name or posix.is_absolute() or windows.drive or '..' in posix.parts:
        raise ValueError('unsafe manifest path: '+name)
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('path escapes package: '+name)
    return path

def expected_deep():
    names=[]
    for ds in DATASETS:
        for model in ('tensor','full','lstm','dual','retain'):
            for seed in SEEDS:
                base=f'{ds}_{model}_seed{seed}'
                names.append('final/'+base)
                if model in ('tensor','full'):names.append('final/'+base+'_mse')
                policies=('importance','pearson') if model in ('tensor','full') else ('importance',) if model in ('dual','retain') else ()
                names.extend('selection/'+base+'_'+policy for policy in policies)
    return names

def check_matrix(names):
    expected=expected_deep()
    if len(names)!=len(expected) or set(names)!=set(expected):
        raise ValueError('incomplete or duplicate 195-run dataset/model/seed/selection matrix')

def verify_files(root,files):
    for name,proof in files.items():
        path=checked_path(root,name)
        if not path.is_file() or path.stat().st_size!=proof['bytes'] or sha(path)!=proof['sha256']:
            raise ValueError('retained artifact integrity mismatch: '+name)
    return len(files)

def required_public_files():
    return {'README.md','CONTRIBUTING.md','requirements.txt','requirements-windows-lock.txt','sources.lock.json',
            'configs/reproduction.json','configs/statistical_audit_policy.json',
            'code/final_package.py','code/verify_final.py','code/rebuild_reports.py','code/windows_compat.py',
            'code/sitecustomize.py','code/setup_project.py','code/prepare_data.py','code/common.py',
            'code/repro/models.py','code/repro/engine.py','code/repro/lifecycle.py','code/repro/verify.py',
            'docs/FINAL_AUDIT.md','docs/RUNNING.md','docs/SOURCE_AUDIT.md',
            'reports/复现实验报告.md','reports/seed_metrics.csv','reports/summary_metrics.csv',
            'reports/variable_selection.csv','reports/uncertainty.csv','reports/interpretation_stability.csv',
            'reports/data_audit.csv','reports/paper_comparison.csv','reports/figures/forecast_errors.png',
            'reports/figures/importance_temporal.png','reports/evidence/hpo_trials.json',
            'reports/evidence/original_deep_audit_66754a81490f.json','reports/evidence/source_artifact_audit.json',
            'reports/evidence/original_strict_final_coordinate_audit.json','reports/evidence/final_verification.json'}

def validate_inventory_contract(files,public=False):
    required=required_public_files()
    if not public:
        for name in expected_deep():
            for file in ['best.pt','config.json','result.json','predictions.csv','history.csv','importance.csv',
                         'importance_history.json','original_artifacts.json','original_complete.json']:
                required.add('output/reproduction/'+name+'/'+file)
        for ds in DATASETS:
            required.update([f'data/processed/{ds}.npz',f'data/processed/{ds}.json'])
            for kind in KINDS:
                base=f'output/reproduction/classic/{ds}_{kind}/'
                for file in ['config.json','result.json','original_artifacts.json','original_complete.json']:
                    required.add(base+file)
                for i in range(5 if kind in ('rf','xgt') else 1):required.add(base+f'predictions_{i}.csv')
                if kind in ('strx','arimax'):required.add(base+'fit_details.json')
    missing=required-set(files)
    if missing:raise ValueError('final seal omits required retained paths: '+str(sorted(missing)))

def retain_deep_file(name):
    return name in {'best.pt','config.json','result.json','predictions.csv','history.csv',
                    'importance.csv','importance_history.json','temporal.npz'}

def restore_deep_predictions(csv_scaled,scale,minimum):
    import numpy as np
    # engine.py exports float32 scaled values with shortest decimal formatting,
    # but explicitly casts them to float64 before inverse scaling.
    return np.asarray(csv_scaled,dtype=np.float32).astype(np.float64)*float(scale)+float(minimum)

def frozen_statistical_grid(data,info,columns):
    """Replay the original parameter design, never rerun a platform-dependent QR.

    Saved coefficients have the identity/order of the training-selected columns.
    A new rank-revealing QR can choose an equivalent dummy basis whose physical
    coefficients differ. Reusing coefficients with that new basis is incorrect.
    """
    import numpy as np
    columns=list(columns);n=len(info['features'])
    if len(columns)!=len(set(columns)) or any(type(i) is not int or i<0 or i>=n or i==info['target_column'] for i in columns):
        raise ValueError('invalid saved exogenous column identities')
    first=int(data['train_idx'][0]);size=int(data['test_idx'][-1])+1-first
    cut=int(data['val_idx'][0])-first
    x=np.full((size,len(columns)),np.nan);y=np.full(size,np.nan)
    for split in ['train','val','test']:
        ix=data[split+'_idx']-first;y[ix]=data[split+'_y']
        x[ix]=data[split+'_x'][:,-1,:][:,columns]
    fill=np.nanmean(x[:cut],axis=0)
    x=np.where(np.isfinite(x),x,fill)
    return x,y,first,cut

def check_hpo_group(trials,selected):
    if len(trials)!=selected['candidate_count'] or len({x['run'] for x in trials})!=len(trials):
        raise ValueError('incomplete or duplicate hyperparameter search group')
    valid=[]
    for trial in trials:
        if trial['status']=='completed':
            value=trial['val_rmse_scaled']
            if value is None or not math.isfinite(value):raise ValueError('nonfinite completed HPO score')
            valid.append(trial)
        elif trial['status']!='nonfinite_rejected' or trial['val_rmse_scaled'] is not None:
            raise ValueError('unsupported HPO outcome')
    if not valid:raise ValueError('no valid HPO candidate')
    winner=min(valid,key=lambda x:x['val_rmse_scaled'])
    if winner['run']!=selected['run'] or winner['val_rmse_scaled']!=selected['val_rmse_scaled']:
        raise ValueError('hyperparameter selection differs from minimum validation RMSE')

def check_statistical_certificate(certificate,proofs):
    expected={f'{ds}_{kind}' for ds in DATASETS for kind in ['strx','arimax']}
    if certificate.get('status')!='PASS' or certificate.get('training_only') is not True or set(certificate.get('groups',{}))!=expected or set(proofs)!=expected:
        raise ValueError('incomplete strict statistical certificate')
    for key in expected:
        row=certificate['groups'][key]
        if row.get('status')!='PASS':raise ValueError('rejected statistical certificate endpoint: '+key)
        for field in ['result_sha256','manifest_sha256','parameters_sha256','data_sha256','initialization']:
            if row.get(field)!=proofs[key][field]:raise ValueError('strict statistical certificate binding changed: '+key+'/'+field)
        for field in ['score','boundary_violation']:
            value=row.get(field)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1e-6:
                raise ValueError('strict statistical threshold failed: '+key+'/'+field)
        directions=row.get('directions',[])
        if len(directions)!=3 or any(item.get('pass') is not True for item in directions):
            raise ValueError('strict statistical directional certificate failed: '+key)

def manifest_files(root,paths):
    root=Path(root).resolve();result={}
    for path in sorted(map(Path,paths)):
        name=path.resolve().relative_to(root).as_posix()
        result[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    return result

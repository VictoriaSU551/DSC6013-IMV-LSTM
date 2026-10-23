"""Durable single-GPU experiment scheduler; safe to restart, guarded by flock."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .engine import ROOT,atomic_json,load_data,sha,fingerprint_config
from .lifecycle import is_complete,validate_config
from .audited_repairs import resolve_classical,verify_repair,is_strict_complete

BASE=ROOT.parent
OUT=ROOT/'output/reproduction'
CONFIG=json.loads((ROOT/'configs/reproduction.json').read_text())
TOTAL=363


def progress(stage,current=None):
    # Search units count valid or explicitly rejected numerical candidates; final fits require success.
    hpo_done=sum(1 for p in (OUT/'hpo').glob('*/complete.json')) if (OUT/'hpo').exists() else 0
    hpo_done+=sum(1 for p in (OUT/'hpo').glob('*/invalid_candidate.json')) if (OUT/'hpo').exists() else 0
    final_done=sum(1 for p in (OUT/'final').glob('*/complete.json')) if (OUT/'final').exists() else 0
    select_done=sum(1 for p in (OUT/'selection').glob('*/complete.json')) if (OUT/'selection').exists() else 0
    classic_done=sum(int(is_strict_complete(ds,kind)) for ds in CONFIG['datasets'] for kind in ['enet','rf','xgt','strx','arimax'])
    initial=int((OUT/'data_audit.json').exists())+int((OUT/'method_verification.json').exists())
    end=int((OUT/'delivery_ready.json').exists())
    done=hpo_done+final_done+select_done+classic_done+initial+end
    atomic_json(BASE/'progress.json',{'stage':stage,'current':current,'status':'running','completed_units':done,
                'total_units':TOTAL,'percent':round(100*done/TOTAL,2),'hpo':hpo_done,'final':final_done,
                'selection':select_done,'classical_groups':classic_done,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def wait_gpu():
    while True:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not output: return
        progress('waiting_for_free_gpu',output)
        time.sleep(60)


def run_unit(job,category,name,invalid_ok=False):
    folder=OUT/category/name; folder.mkdir(parents=True,exist_ok=True)
    if invalid_ok and (folder/'invalid_candidate.json').exists():
        validate_config(folder/'config.json',fingerprint_config(job['config'],ROOT/'data/processed'/f"{job['config']['dataset']}.npz",job['final']))
        return None
    job={**job,'folder':str(folder)}
    task=OUT/'jobs'/f'{category}_{name}.json'; atomic_json(task,job)
    progress(category,name)
    attempts=2
    for attempt in range(attempts):
        if job['type']=='deep' and not is_complete(folder): wait_gpu()
        log=BASE/'logs'/f'{category}_{name}.log'
        with log.open('a') as f:
            proc=subprocess.Popen([sys.executable,'-B','-m','repro.worker',str(task)],stdout=f,stderr=subprocess.STDOUT,
                                   cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'code')})
            atomic_json(OUT/'active_worker.json',{'pid':proc.pid,'category':category,'name':name,'job':str(task),'log':str(log)})
            while proc.poll() is None:
                # Status lives in the worker folder; this doesn't emit user-facing chatter.
                time.sleep(30); progress(category,name)
        if proc.returncode==0:
            result=json.loads((folder/'result.json').read_text()); progress(category,name); return result
        failure=json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else {'error_type':'ProcessExit','error':str(proc.returncode)}
        if invalid_ok and failure['error_type']=='FloatingPointError':
            atomic_json(folder/'invalid_candidate.json',failure); progress(category,name); return None
        if attempt+1<attempts: time.sleep(10)
    raise RuntimeError(f'unit failed, evidence preserved: {category}/{name}: {failure}')


def cfg(dataset,model,params,seed,search=False,objective=None,selected=None,selection='all'):
    units,lr,dropout,l2=params
    c={'dataset':dataset,'model':model,'objective':objective or ('em' if model in ('full','tensor') else 'mse'),
       'units':units,'lr':lr,'dropout':dropout,'l2':l2,'seed':seed,
       'epochs':CONFIG['hpo_epochs'] if search else CONFIG['epochs'],
       'patience':CONFIG['hpo_patience'] if search else CONFIG['patience'],
       'batch_size':CONFIG['batch_size'],'threads':CONFIG['threads'],'device':'cuda',
       'sigma_floor':CONFIG['sigma_floor'],'selection':selection}
    if selected is not None: c['selected']=selected
    return c


def search(dataset,model):
    path=OUT/'selected'/f'{dataset}_{model}.json'
    candidates=CONFIG['imv_candidates'] if model in ('full','tensor') else CONFIG['baseline_candidates']
    records=[]
    for i,params in enumerate(candidates):
        name=f'{dataset}_{model}_{i:02d}'
        result=run_unit({'type':'deep','config':cfg(dataset,model,params,CONFIG['search_seed'],True),'final':False},'hpo',name,True)
        if result is not None: records.append({'params':params,'val_rmse_scaled':result['val_rmse_scaled'],'run':name})
    if not records: raise RuntimeError(f'no valid HPO candidate {dataset}/{model}')
    winner=min(records,key=lambda r:r['val_rmse_scaled'])
    atomic_json(path,{**winner,'all_valid':records,'selection_criterion':'validation RMSE, never test','candidate_count':len(candidates)})
    return winner['params']


def select_features(dataset,model,seed,selection):
    d,info=load_data(ROOT/'data/processed'/f'{dataset}.npz')
    n=len(info['features']); k=math.ceil(n/2)
    if selection=='pearson':
        # Forecast-aligned Pearson: flattened lag values paired with repeated next-step training y.
        # Average absolute correlation over lags, not labels from validation/test.
        y=d['train_y'].astype(float); scores=[]
        for v in range(n):
            correlations=[]
            for lag in range(info['depth']):
                x=d['train_x'][:,lag,v].astype(float)
                correlations.append(abs(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else 0.)
            scores.append(float(np.mean(correlations)))
    else:
        rank=pd.read_csv(OUT/'final'/f'{dataset}_{model}_seed{seed}'/'importance.csv')
        scores=rank['importance'].to_numpy()
    chosen=sorted(np.argsort(-np.nan_to_num(scores),kind='stable')[:k].tolist())
    atomic_json(OUT/'selections'/f'{dataset}_{model}_seed{seed}_{selection}.json',
                {'indices':chosen,'features':[info['features'][i] for i in chosen],'scores':list(map(float,scores)),
                 'source':'training only','dual_decoder_target_retained':model=='dual'})
    return chosen


def audit():
    for ds in CONFIG['datasets']:
        p=ROOT/'data/processed'/f'{ds}.npz'; d,info=load_data(p)
        for a,b in [('train','val'),('val','test')]:
            assert d[a+'_idx'].max()<d[b+'_idx'].min()
        assert all(np.isfinite(d[s+'_x']).all() and np.isfinite(d[s+'_y']).all() for s in ['train','val','test'])
    atomic_json(OUT/'data_audit.json',{'status':'PASS','datasets':CONFIG['datasets'],'source_lock_sha256':sha(ROOT/'sources.lock.json')})
    evidence=BASE/'logs/tests.log'
    if not evidence.exists() or 'failed' in evidence.read_text().lower() or 'passed' not in evidence.read_text().lower():
        raise RuntimeError('method tests must pass before suite starts')
    atomic_json(OUT/'method_verification.json',{'status':'PASS','test_log_sha256':sha(evidence)})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'suite.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('suite already running; no duplicate launched')
    atomic_json(OUT/'supervisor.json',{'pid':os.getpid(),'config_sha256':sha(ROOT/'configs/reproduction.json')})
    try:
        audit()
        for dataset in CONFIG['datasets']:
            winners={model:search(dataset,model) for model in ['tensor','full','lstm','dual','retain']}
            for model in ['tensor','full','lstm','dual','retain']:
                for seed in CONFIG['seeds']:
                    name=f'{dataset}_{model}_seed{seed}'
                    run_unit({'type':'deep','config':cfg(dataset,model,winners[model],seed),'final':True},'final',name)
                    if model in ('tensor','full'):
                        run_unit({'type':'deep','config':cfg(dataset,model,winners[model],seed,objective='mse'),'final':True},
                                 'final',name+'_mse')
            for model in ['tensor','full','dual','retain']:
                policies=['importance','pearson'] if model in ('tensor','full') else ['importance']
                for selection in policies:
                    for seed in CONFIG['seeds']:
                        selected=select_features(dataset,model,seed,selection)
                        c=cfg(dataset,model,winners[model],seed,selected=selected,selection=selection)
                        run_unit({'type':'deep','config':c,'final':True},'selection',f'{dataset}_{model}_seed{seed}_{selection}')
        # Classical optimizer problems must not prevent all public-data core runs.
        for dataset in CONFIG['datasets']:
            for model in ['enet','rf','xgt','strx','arimax']:
                folder=resolve_classical(dataset,model)
                if is_complete(folder) and not is_strict_complete(dataset,model):
                    progress('strict_audit_pending',f'{dataset}_{model}');continue
                if folder.parent.name=='repairs':
                    verify_repair(folder);progress('classic',f'{dataset}_{model}')
                else:run_unit({'type':'classic','dataset':dataset,'model':model,'seeds':CONFIG['seeds']},'classic',f'{dataset}_{model}')
        unresolved=[f'{ds}_{kind}' for ds in CONFIG['datasets'] for kind in ['enet','rf','xgt','strx','arimax'] if not is_strict_complete(ds,kind)]
        if unresolved: raise RuntimeError('classical groups need convergence review: '+str(unresolved))
        progress('verification_and_report')
        subprocess.run([sys.executable,'-B','-m','repro.verify'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-B','-m','repro.report'],cwd=ROOT,check=True)
        atomic_json(OUT/'delivery_ready.json',{'status':'awaiting_agent_review_and_publish','report':str(BASE/'delivery')})
        progress('awaiting_agent_review_and_publish')
    except Exception as exc:
        import traceback
        atomic_json(BASE/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
        state=json.loads((BASE/'progress.json').read_text());state['status']='needs_repair';state['error']=str(exc)
        atomic_json(BASE/'progress.json',state)
        raise

if __name__=='__main__': main()

"""Verify the retained final package without restoring discarded training files.

The default rechecks every retained byte, the prior inference certificate's
bindings, all saved prediction metrics and all experiment/selection cells.
--replay additionally reloads selected checkpoints on the current CPU/GPU.
--rebuild-data compares raw-data reconstruction in memory, never overwrites data.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import time
from windows_compat import install
install()
import numpy as np
import pandas as pd
import torch
import final_package as p
from repro.engine import load_data
from repro.models import build_model
from repro.verify import verify_run,verify_classical_alignment,verify_classical_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/reproduction'
EVIDENCE=ROOT/'reports/evidence'

def require(value,message):
    if not value:raise ValueError(message)

def metric(y,pred):
    error=np.asarray(pred,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return {'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error)))}

def check_original(folder):
    original=p.read(folder/'original_artifacts.json');marker=p.read(folder/'original_complete.json')
    require(p.sha(folder/'original_artifacts.json')==marker['manifest_sha256'],'original manifest changed')
    require(p.sha(folder/'result.json')==marker['result_sha256'],'original result changed')
    require(marker['status']=='completed','unaccepted final run')
    for name,want in original.items():
        file=folder/name
        if folder.parent.name in ('final','selection') and p.retain_deep_file(name):
            require(file.is_file(),'missing required retained deep artifact: '+str(file))
        if file.exists():require(p.sha(file)==want,'retained original artifact changed: '+str(file))
    return original

def certificate_bindings(manifest):
    audit=p.read(EVIDENCE/'original_deep_audit_66754a81490f.json')
    require(audit['status']=='PASS' and audit['verified']==195 and audit['expected']==195,'incomplete original inference certificate')
    mapping={r['run']:r for r in audit['runs']}
    require(len(mapping)==195 and len(audit['runs'])==195,'duplicate original inference rows')
    require(audit['function_sha256']==hashlib.sha256(inspect.getsource(verify_run).encode()).hexdigest(),'original inference function changed')
    for name,want in audit['source_sha256'].items():require(p.sha(ROOT/'code/repro'/name)==want,'inference source changed: '+name)
    for name,want in audit['data_sha256'].items():require(p.sha(ROOT/'data/processed'/name)==want,'inference data changed: '+name)
    for name in manifest['deep_runs']:
        folder=OUT/name;row=mapping.get(folder.name)
        require(row is not None and row['status']=='PASS','missing original inference row: '+name)
        for file,field in [('config.json','config.json'),('result.json','result.json'),
                           ('original_artifacts.json','artifacts.json'),('original_complete.json','complete.json')]:
            require(p.sha(folder/file)==row['proof'][field],'original inference proof changed: '+name+'/'+file)
        check_original(folder)
    return len(mapping)

def statistical_certificate_bindings():
    certificate=p.read(EVIDENCE/'original_strict_final_coordinate_audit.json');proofs={}
    for ds in p.DATASETS:
        for kind in ['strx','arimax']:
            key=f'{ds}_{kind}';folder=OUT/'classic'/key;details=p.read(folder/'fit_details.json')
            proofs[key]={'result_sha256':p.sha(folder/'result.json'),'manifest_sha256':p.sha(folder/'original_artifacts.json'),
                         'parameters_sha256':p.sha(folder/'fit_details.json'),'data_sha256':p.sha(ROOT/f'data/processed/{ds}.npz'),
                         'initialization':details['initialization']}
    p.check_statistical_certificate(certificate,proofs)
    for name,want in certificate['dependencies'].items():require(p.sha(ROOT/'code/repro'/name)==want,'strict statistics dependency changed')
    for name,want in certificate['used_sources_sha256'].items():require(p.sha(p.checked_path(ROOT,name))==want,'strict statistics source changed')
    for ds,want in certificate['data_metadata_sha256'].items():require(p.sha(ROOT/f'data/processed/{ds}.json')==want,'strict statistics metadata changed')
    return len(proofs)

def check_deep_saved(folder,data):
    cfg=p.read(folder/'config.json');result=p.read(folder/'result.json')
    require(result['status']=='completed','incomplete deep result')
    require(cfg['data_sha256']==p.sha(ROOT/f"data/processed/{cfg['dataset']}.npz"),'deep data binding')
    for name,want in cfg['source_sha256'].items():require(p.sha(ROOT/'code/repro'/name)==want,'deep source binding')
    frame=pd.read_csv(folder/'predictions.csv')
    np.testing.assert_array_equal(frame.target_row,data['test_idx'])
    np.testing.assert_allclose(frame.y_true,data['test_raw_y'],rtol=0,atol=1e-8)
    np.testing.assert_array_equal(frame.y_true_scaled.to_numpy(dtype=np.float32),data['test_y'])
    scaled=frame.y_pred_scaled.to_numpy(dtype=np.float32)
    raw=p.restore_deep_predictions(scaled,data['y_scale'],data['y_min'])
    np.testing.assert_allclose(frame.y_pred.to_numpy(),raw,rtol=1e-14,atol=1e-12)
    for unit,y,pred in [('raw',data['test_raw_y'],raw),('scaled',data['test_y'],scaled)]:
        values=metric(y,pred)
        for key,value in values.items():require(math.isclose(value,result['test_'+unit][key],rel_tol=1e-9,abs_tol=1e-10),'saved deep metric: '+str(folder)+' '+unit+'/'+key)
    history=pd.read_csv(folder/'history.csv');winner=history.loc[history.val_rmse_scaled.idxmin()]
    require(int(winner.epoch)==result['best_epoch'],'best validation epoch mismatch')
    require(math.isclose(winner.val_rmse_scaled,result['val_rmse_scaled'],rel_tol=1e-10,abs_tol=1e-12),'validation score mismatch')
    rank=pd.read_csv(folder/'importance.csv')
    require(np.isfinite(rank.importance).all() and (rank.importance>=0).all() and abs(rank.importance.sum()-1)<1e-5,'invalid importance')
    return result

def replay_deep(folder,device):
    cfg=p.read(folder/'config.json');result=p.read(folder/'result.json')
    data,info=load_data(ROOT/f"data/processed/{cfg['dataset']}.npz",cfg.get('selected'))
    model=build_model(cfg['model'],len(info['features']),cfg['units'],info['depth'],info['target_column'],cfg['dropout'],cfg.get('sigma_floor',.001)).to(device).eval()
    checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=False)
    require(checkpoint['epoch']==result['best_epoch'],'selected checkpoint epoch mismatch')
    model.load_state_dict(checkpoint['model']);prediction=[]
    qsum=np.zeros(len(info['features']));asum=None;count=0
    with torch.no_grad():
        for start in range(0,len(data['test_y']),256):
            x=torch.tensor(data['test_x'][start:start+256],device=device)
            out=model(x,target_history=torch.tensor(data['test_target_history'][start:start+256],device=device)) if cfg['model']=='dual' else model(x)
            prediction.extend(out['prediction'].cpu().numpy().tolist())
        if cfg['objective']=='em':
            for start in range(0,len(data['train_y']),256):
                x=torch.tensor(data['train_x'][start:start+256],device=device);y=torch.tensor(data['train_y'][start:start+256],device=device)
                out=model(x);joint=torch.log_softmax(out['logits'],-1)-torch.log(out['sigma'])-.5*((y[:,None]-out['mu'])/out['sigma'])**2
                qsum+=torch.softmax(joint,-1).double().sum(0).cpu().numpy();count+=len(y)
                if 'alpha' in out:
                    value=out['alpha'].double().sum(0).cpu().numpy();asum=value if asum is None else asum+value
    frame=pd.read_csv(folder/'predictions.csv');prediction=np.asarray(prediction)
    np.testing.assert_allclose(prediction,frame.y_pred_scaled,rtol=1e-5,atol=2e-6)
    if count:
        np.testing.assert_allclose(qsum/count,pd.read_csv(folder/'importance.csv').importance,rtol=2e-4,atol=2e-5)
        if asum is not None:
            with np.load(folder/'temporal.npz') as temporal:
                np.testing.assert_allclose(asum/count,temporal[temporal.files[0]],rtol=2e-4,atol=2e-5)
    return {'run':folder.name,'status':'PASS','max_abs_scaled_prediction_difference':float(max(abs(prediction-frame.y_pred_scaled.to_numpy())))}

def replay_statistical_fixed(data,info,details,kind):
    from repro.repairs import model
    x,y,first,cut=p.frozen_statistical_grid(data,info,details['exogenous_original_indices'])
    require((first,cut)==(details['first'],details['cut']),'saved statistical time-grid mismatch')
    training=model(y[:cut],x[:cut],info,kind,details['initialization'])
    require(training.param_names==details['parameter_names'],'saved statistical parameter identity mismatch')
    params=np.asarray(details['parameters'],dtype=float)
    ll=float(training.loglike(params))
    combined=model(y,x,info,kind,details['initialization'])
    filtered=combined.filter(params,cov_type='none',low_memory=False)
    predictions=np.asarray(filtered.predict(start=cut,end=len(y)-1,dynamic=False))
    return predictions[data['test_idx']-first-cut],ll

def check_search_and_selection(data):
    trials=p.read(EVIDENCE/'hpo_trials.json');require(len(trials)==150,'missing HPO trials')
    for ds in p.DATASETS:
        y=data[ds]['train_y'].astype(float);x=data[ds]['train_x'].astype(float);pearson=[]
        for v in range(x.shape[-1]):
            corrs=[abs(np.corrcoef(x[:,lag,v],y)[0,1]) if x[:,lag,v].std()>0 and y.std()>0 else 0. for lag in range(x.shape[1])]
            pearson.append(float(np.mean(corrs)))
        del x
        for model in ('tensor','full','lstm','dual','retain'):
            prefix=f'{ds}_{model}_';group=[t for t in trials if t['run'].startswith(prefix)]
            chosen=p.read(OUT/'selected'/f'{ds}_{model}.json');p.check_hpo_group(group,chosen)
            for trial in group:
                result=trial['result'];hist=trial['history'];best=min(hist,key=lambda x:float(x['val_rmse_scaled']))
                require('test_raw' not in result,'test evaluation in HPO')
                require(result['best_epoch']==int(best['epoch']) and math.isclose(result['val_rmse_scaled'],float(best['val_rmse_scaled']),rel_tol=1e-10),'HPO best epoch mismatch')
                require(trial['config']['final'] is False,'HPO configuration marked final')
        for model in ('tensor','full','dual','retain'):
            policies=('importance','pearson') if model in ('tensor','full') else ('importance',)
            n=len(p.read(ROOT/f'data/processed/{ds}.json')['features']);k=math.ceil(n/2)
            for seed in p.SEEDS:
                for policy in policies:
                    name=f'{ds}_{model}_seed{seed}_{policy}';selection=p.read(OUT/'selections'/f'{name}.json')
                    if policy=='importance':scores=pd.read_csv(OUT/'final'/f'{ds}_{model}_seed{seed}'/'importance.csv').importance.to_numpy()
                    else:scores=pearson
                    selected=sorted(np.argsort(-np.nan_to_num(scores),kind='stable')[:k].tolist())
                    require(selected==selection['indices'],'training-only feature ranking mismatch: '+name)
                    require(p.read(OUT/'selection'/name/'config.json')['selected']==selected,'selected training input mismatch')
    return len(trials)

def rebuild_data():
    from prepare_data import read_raw,make_windows,scale_splits
    rows=[]
    for ds in p.DATASETS:
        raw,features,target,cuts,segments,files=read_raw(ds)
        x,y,idx=make_windows(raw,target,10,segments);rebuilt=scale_splits(x,y,idx,cuts)
        with np.load(ROOT/f'data/processed/{ds}.npz') as saved:
            require(set(saved.files)==set(rebuilt),'processed array key mismatch')
            for key,value in rebuilt.items():np.testing.assert_array_equal(value,saved[key])
        rows.append({'dataset':ds,'status':'PASS','arrays':len(rebuilt)})
        del raw,x,y,rebuilt;gc.collect()
    return rows

def check_report(rows):
    saved=pd.read_csv(ROOT/'reports/seed_metrics.csv')
    require(len(saved)==len(rows),'report seed row coverage')
    for row in rows:
        match=saved[(saved.run==row['run']) & ((saved.seed==row['seed']) if row['seed'] is not None else saved.seed.isna())]
        require(len(match)==1,'missing/duplicate report row: '+row['run'])
        for key in ['dataset','model','objective','selection']:
            require(match.iloc[0][key]==row[key],'seed report classification mismatch: '+row['run']+'/'+key)
        for key in ['rmse_raw','mae_raw','rmse_scaled','mae_scaled']:
            require(math.isclose(float(match.iloc[0][key]),row[key],rel_tol=1e-12,abs_tol=1e-12),'seed report metric mismatch')
    summary=pd.read_csv(ROOT/'reports/summary_metrics.csv');frame=pd.DataFrame(rows)
    for keys,group in frame.groupby(['dataset','model','objective','selection']):
        mask=np.ones(len(summary),dtype=bool)
        for key,value in zip(['dataset','model','objective','selection'],keys):mask &= summary[key]==value
        match=summary[mask];require(len(match)==1,'missing summary group')
        require(int(match.iloc[0].n_runs)==len(group),'summary run count mismatch')
        for metric_name in ['rmse_raw','mae_raw','rmse_scaled','mae_scaled']:
            vals=group[metric_name].to_numpy();sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            for suffix,value in [('mean',vals.mean()),('sd',sd),('se',sd/math.sqrt(len(vals)))]:
                require(math.isclose(float(match.iloc[0][metric_name+'_'+suffix]),value,rel_tol=1e-11,abs_tol=1e-12),'summary statistic mismatch')
    require(len(summary)==frame.groupby(['dataset','model','objective','selection']).ngroups,'extra summary groups')
    return {'seed_rows':len(saved),'summary_groups':len(summary)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',action='store_true',help='Check the published code/report subset without claiming local artifact verification')
    parser.add_argument('--rebuild-data',action='store_true')
    parser.add_argument('--replay',choices=['none','representative','all'],default='none')
    parser.add_argument('--replay-statistics',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();start=time.monotonic();torch.set_num_threads(2)
    manifest=p.read(EVIDENCE/'final_manifest.json');p.check_matrix(manifest['deep_runs'])
    expected_classic={f'{ds}_{kind}' for ds in p.DATASETS for kind in p.KINDS}
    require(set(manifest['classical_groups'])==expected_classic and len(manifest['classical_groups'])==15,'classical matrix mismatch')
    public={name:proof for name,proof in manifest['files'].items() if not name.startswith(('data/','upstream/','output/','code/vendor/'))}
    p.validate_inventory_contract(manifest['files'],public=args.public)
    count=p.verify_files(ROOT,public if args.public else manifest['files'])
    result={'status':'PASS','scope':'published_code_and_reports_only' if args.public else 'retained_public_data_experiments',
            'verified_files':count,'full_paper_reproduction':False,'runtime':{'python':platform.python_version(),'torch':torch.__version__}}
    if not args.public:
        lock=p.read(ROOT/'sources.lock.json')
        for asset in lock['assets']:require(p.sha(ROOT/asset['path'])==asset['sha256'],'source asset changed')
        result['prior_checkpoint_inference_certificates_revalidated']=certificate_bindings(manifest)
        result['prior_strict_statistical_certificates_revalidated']=statistical_certificate_bindings()
        data={ds:load_data(ROOT/f'data/processed/{ds}.npz')[0] for ds in p.DATASETS};rows=[]
        for name in manifest['deep_runs']:
            folder=OUT/name;cfg=p.read(folder/'config.json');r=check_deep_saved(folder,data[cfg['dataset']])
            rows.append({**{k:r[k] for k in ['dataset','model','objective','selection','seed']},'run':folder.name,
                         **{metric_name+'_'+unit:r['test_'+unit][metric_name] for unit in ['raw','scaled'] for metric_name in ['rmse','mae']}})
        result['hpo_trials']=check_search_and_selection(data)
        statistics=[]
        for ds in p.DATASETS:
            for kind in p.KINDS:
                folder=OUT/'classic'/f'{ds}_{kind}';check_original(folder);r=p.read(folder/'result.json')
                cfg=p.read(folder/'config.json')
                for name,want in cfg['source_sha256'].items():require(p.sha(ROOT/'code/repro'/name)==want,'classical source binding changed')
                require(r['data_sha256']==p.sha(ROOT/f'data/processed/{ds}.npz'),'classical data binding')
                actual=[row['seed'] for row in r['runs']];require(actual==(list(p.SEEDS) if kind in ('rf','xgt') else [None]),'classical seeds mismatch')
                for i,row in enumerate(r['runs']):
                    frame=pd.read_csv(folder/f'predictions_{i}.csv');verify_classical_alignment(frame,data[ds]);verify_classical_metrics(frame,data[ds],row,kind)
                    rows.append({'dataset':ds,'model':kind,'objective':'classical','selection':'all','seed':row['seed'],'run':f'{ds}_{kind}',
                                 **{metric_name+'_'+unit:row['test_'+unit][metric_name] for unit in ['raw','scaled'] for metric_name in ['rmse','mae']}})
                if args.replay_statistics and kind in ('strx','arimax'):
                    details=p.read(folder/'fit_details.json');d,info=load_data(ROOT/f'data/processed/{ds}.npz')
                    prediction,ll=replay_statistical_fixed(d,info,details,kind)
                    saved=pd.read_csv(folder/'predictions_0.csv').y_pred_scaled.to_numpy()
                    np.testing.assert_allclose(prediction,saved,rtol=1e-7,atol=1e-8)
                    require(math.isclose(ll,details['train_loglike'],rel_tol=1e-8,abs_tol=1e-6),'cross-platform fixed-parameter likelihood mismatch')
                    statistics.append({'group':f'{ds}_{kind}','status':'PASS','max_abs_scaled_prediction_difference':float(max(abs(prediction-saved)))})
                    print(f'FIXED_STATISTICS_REPLAY {ds}/{kind} PASS',flush=True)
        result['deep_saved_metrics']=195;result['classical_groups']=15;result['report_reconciliation']=check_report(rows)
        if args.rebuild_data:result['raw_data_rebuild']=rebuild_data()
        result['current_checkpoint_replays']=[]
        if args.replay!='none':
            names=manifest['deep_runs'] if args.replay=='all' else [name for name in manifest['deep_runs'] if name.startswith('final/') and name.endswith('seed11')]
            for index,name in enumerate(names):
                row=replay_deep(OUT/name,args.device);result['current_checkpoint_replays'].append(row)
                print(f'CHECKPOINT_REPLAY {index+1}/{len(names)} {name} PASS',flush=True)
        result['current_fixed_parameter_statistics_replays']=statistics
    result['seconds']=time.monotonic()-start
    if not args.public:
        destination=ROOT/'output/final_verification.json';destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""Independent checkpoint inference and metric/selection verification."""
import json
import argparse
import hashlib
import inspect
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .engine import ROOT,atomic_json,load_data,sha
from .models import build_model
from .lifecycle import is_complete
from .audited_repairs import resolve_classical,verify_repair,require_strict_complete

OUT=ROOT/'output/reproduction'
CFG=json.loads((ROOT/'configs/reproduction.json').read_text())


def expected_runs():
    result=[]
    for dataset in CFG['datasets']:
        for model in ['tensor','full','lstm','dual','retain']:
            for seed in CFG['seeds']:
                name=f'{dataset}_{model}_seed{seed}'
                result.append(OUT/'final'/name)
                if model in ('tensor','full'): result.append(OUT/'final'/(name+'_mse'))
                policies=['importance','pearson'] if model in ('tensor','full') else ['importance'] if model in ('dual','retain') else []
                for policy in policies: result.append(OUT/'selection'/(name+'_'+policy))
    return result


def verify_run(folder,device='cuda'):
    cfg=json.loads((folder/'config.json').read_text()); r=json.loads((folder/'result.json').read_text())
    assert r['status']=='completed' and is_complete(folder)
    assert sha(ROOT/'data/processed'/f"{cfg['dataset']}.npz")==cfg['data_sha256']
    for name,digest in json.loads((folder/'artifacts.json').read_text()).items(): assert sha(folder/name)==digest,(folder,name)
    history=pd.read_csv(folder/'history.csv')
    idx=int(history['val_rmse_scaled'].idxmin()); row=history.iloc[idx]
    assert int(row['epoch'])==r['best_epoch']
    assert math.isclose(float(row['val_rmse_scaled']),r['val_rmse_scaled'],rel_tol=1e-10,abs_tol=1e-12)
    d,info=load_data(ROOT/'data/processed'/f"{cfg['dataset']}.npz",cfg.get('selected'))
    model=build_model(cfg['model'],len(info['features']),cfg['units'],info['depth'],info['target_column'],cfg['dropout'],cfg.get('sigma_floor',.001)).to(device).eval()
    ck=torch.load(folder/'best.pt',map_location=device,weights_only=False)
    assert ck['epoch']==r['best_epoch']; model.load_state_dict(ck['model'])
    prediction=[]
    with torch.no_grad():
        for start in range(0,len(d['test_y']),256):
            x=torch.tensor(d['test_x'][start:start+256],device=device)
            if cfg['model']=='dual':
                out=model(x,target_history=torch.tensor(d['test_target_history'][start:start+256],device=device))
            else: out=model(x)
            prediction.extend(out['prediction'].cpu().numpy().tolist())
    prediction=np.array(prediction,dtype=float)
    csv=pd.read_csv(folder/'predictions.csv')
    np.testing.assert_array_equal(csv.target_row.to_numpy(),d['test_idx'])
    np.testing.assert_allclose(prediction,csv.y_pred_scaled,rtol=1e-5,atol=2e-6)
    raw=prediction*float(d['y_scale'])+float(d['y_min'])
    np.testing.assert_allclose(csv.y_true,d['test_raw_y'],atol=1e-8)
    for units,y,p in [('raw',d['test_raw_y'],raw),('scaled',d['test_y'],prediction)]:
        error=p-np.asarray(y,dtype=float)
        rmse=float(np.sqrt(np.mean(error**2))); mae=float(np.mean(np.abs(error)))
        assert math.isclose(rmse,r['test_'+units]['rmse'],rel_tol=2e-5,abs_tol=2e-6)
        assert math.isclose(mae,r['test_'+units]['mae'],rel_tol=2e-5,abs_tol=2e-6)
    ranking=pd.read_csv(folder/'importance.csv')
    assert len(ranking)==len(info['features']); assert np.isfinite(ranking.importance).all()
    assert (ranking.importance>=0).all(); assert abs(ranking.importance.sum()-1)<1e-5
    if cfg['objective']=='em':
        # Re-evaluate posterior averaging directly instead of engine.evaluate.
        qsum=np.zeros(len(info['features'])); total=0
        with torch.no_grad():
            for start in range(0,len(d['train_y']),256):
                x=torch.tensor(d['train_x'][start:start+256],device=device)
                y=torch.tensor(d['train_y'][start:start+256],device=device)
                out=model(x)
                joint=torch.log_softmax(out['logits'],-1)-torch.log(out['sigma'])-.5*((y[:,None]-out['mu'])/out['sigma'])**2
                qsum+=torch.softmax(joint,-1).double().sum(0).cpu().numpy(); total+=len(y)
        np.testing.assert_allclose(qsum/total,ranking.importance,rtol=2e-4,atol=2e-5)
    return {'run':folder.name,'status':'PASS','best_epoch':r['best_epoch'],'n':len(prediction)}


def verify_classical_alignment(frame,data):
    assert len(frame)==len(data['test_idx'])
    np.testing.assert_array_equal(frame.target_row.to_numpy(),data['test_idx'])
    np.testing.assert_allclose(frame.y_true,data['test_raw_y'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(frame.y_true_scaled,data['test_y'],rtol=1e-7,atol=1e-8)




def verify_classical_metrics(frame,data,row,kind):
    # pandas exports float32 with its shortest decimal, then defaults to float64
    # on import. Restore the estimator's original dtype before inverse scaling;
    # metrics use canonical labels, never their shortened CSV representation.
    if kind in ('xgt','enet'):dtype=np.float32
    elif kind in ('rf','strx','arimax'):dtype=np.float64
    else:raise ValueError('unsupported estimator precision contract')
    scaled=frame.y_pred_scaled.to_numpy(dtype=dtype)
    raw=scaled*data['y_scale']+data['y_min']
    assert np.isfinite(scaled).all() and np.isfinite(raw).all()
    np.testing.assert_array_equal(frame.y_true_scaled.to_numpy(dtype=data['test_y'].dtype),data['test_y'])
    if dtype is np.float32:np.testing.assert_array_equal(frame.y_pred.to_numpy(dtype=dtype),raw)
    else:np.testing.assert_allclose(frame.y_pred,raw,rtol=1e-12,atol=1e-12)
    for unit,y,pred in [('scaled',data['test_y'],scaled),('raw',data['test_raw_y'],raw)]:
        error=np.asarray(pred,dtype=np.float64)-np.asarray(y,dtype=np.float64)
        for metric,value in [('rmse',float(np.sqrt(np.mean(error**2)))),('mae',float(np.mean(np.abs(error))))]:
            assert np.isclose(value,row['test_'+unit][metric],rtol=1e-9,atol=1e-10),(kind,unit,metric,value,row['test_'+unit][metric])


def check_audit_header(audit,expected,names):
    rows=audit.get('runs',[])
    if audit.get('status')!='PASS' or audit.get('verified')!=len(names) or audit.get('expected')!=len(names):raise ValueError('incomplete independent deep audit')
    for key in ['function_sha256','source_sha256','data_sha256']:
        if audit.get(key)!=expected[key]:raise ValueError('deep audit source/data changed: '+key)
    if len(rows)!=len(names) or set(r['run'] for r in rows)!=set(names) or any(r.get('status')!='PASS' for r in rows):raise ValueError('deep audit run set mismatch')
    return {r['run']:r for r in rows}


def load_deep_audit(path,folders):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise ValueError('deep audit outside project output')
    expected={'function_sha256':hashlib.sha256(inspect.getsource(verify_run).encode()).hexdigest(),
              'source_sha256':{name:sha(ROOT/'code/repro'/name) for name in ['models.py','engine.py','lifecycle.py']},
              'data_sha256':{p.name:sha(p) for p in sorted((ROOT/'data/processed').glob('*')) if p.suffix in ('.npz','.json')}}
    mapping=check_audit_header(json.loads(path.read_text()),expected,[p.name for p in folders])
    for folder in folders:
        proof={name:sha(folder/name) for name in ['config.json','result.json','artifacts.json','complete.json']}
        if mapping[folder.name].get('proof')!=proof or not is_complete(folder):raise ValueError('deep audit artifact changed: '+folder.name)
    return mapping


def main(deep_audit=None):
    torch.set_num_threads(2)
    results=[]; expected=expected_runs()
    assert len(expected)==195
    missing=[str(p) for p in expected if not (p/'result.json').exists()]
    if missing: raise RuntimeError('Missing expected runs: '+str(missing))
    reused=load_deep_audit(deep_audit,expected) if deep_audit else {}
    for folder in expected:
        results.append(reused[folder.name] if reused else verify_run(folder,device=os.environ.get('REPRO_VERIFY_DEVICE','cuda')))
        atomic_json(OUT/'verification_partial.json',{'verified':len(results),'expected':len(expected),'runs':results})
    for dataset in CFG['datasets']:
        for model in ['tensor','full','lstm','dual','retain']:
            selected=json.loads((OUT/'selected'/f'{dataset}_{model}.json').read_text())
            assert selected['val_rmse_scaled']==min(r['val_rmse_scaled'] for r in selected['all_valid'])
        for kind in ['rf','xgt','enet','strx','arimax']:
            folder=require_strict_complete(dataset,kind)
            if folder.parent.name=='repairs':verify_repair(folder)
            result=json.loads((folder/'result.json').read_text())
            assert result['status']=='completed' and is_complete(folder)
            data_path=ROOT/'data/processed'/f'{dataset}.npz'
            assert result['data_sha256']==sha(data_path)
            d,_=load_data(data_path)
            seeds=[row['seed'] for row in result['runs']]
            assert seeds==CFG['seeds'] if kind in ('rf','xgt') else seeds==[None]
            for name,digest in json.loads((folder/'artifacts.json').read_text()).items(): assert sha(folder/name)==digest
            for i,row in enumerate(result['runs']):
                frame=pd.read_csv(folder/f'predictions_{i}.csv')
                verify_classical_alignment(frame,d)
                verify_classical_metrics(frame,d,row,kind)
    for folder in (OUT/'hpo').iterdir():
        if (folder/'result.json').exists():
            r=json.loads((folder/'result.json').read_text())
            assert 'test_raw' not in r and not (folder/'predictions.csv').exists()
    atomic_json(OUT/'verification.json',{'status':'PASS','deep_runs':len(results),'classic_groups':15,
                 'numerical_tolerance':'inference atol2e-6 rtol1e-5 (different batch size)',
                 'deep_audit_sha256':sha(Path(deep_audit)) if deep_audit else None,
                 'classical_precision':'restore float32 xgt/enet; canonical labels; unchanged metric rtol1e-9 atol1e-10; inverse-scale roundtrip checked','runs':results})
    print('INDEPENDENT_VERIFICATION_PASS',len(results),'deep runs and 15 classical groups')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deep-audit');args=parser.parse_args();main(args.deep_audit)

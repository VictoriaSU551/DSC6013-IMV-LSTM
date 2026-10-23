"""Provenance-bound acceptance and fixed-parameter replay of statistical repairs.

Original failed fits are immutable. A sidecar is accepted only after training
likelihood, invertible QR mapping, causal replay and all artifact hashes pass.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .engine import ROOT,sha,atomic_json,load_data,metrics
from .lifecycle import is_complete,mark_complete,run_lock,validate_config

OUT=ROOT/'output/reproduction'


def sources():
    return {name:sha(Path(__file__).with_name(name)) for name in ['repairs.py','engine.py','lifecycle.py','classical.py']}


def runtime():
    return {'python':sys.version.split()[0],**{name:importlib.metadata.version(name) for name in ['numpy','scipy','statsmodels']}}


def grid(data,info):
    exo=[i for i in range(len(info['features'])) if i!=info['target_column']]
    first=int(data['train_idx'][0]);size=int(data['test_idx'][-1])+1-first;cut=int(data['val_idx'][0])-first
    x=np.full((size,len(exo)),np.nan);y=np.full(size,np.nan)
    for split in ['train','val','test']:
        ix=data[split+'_idx']-first;y[ix]=data[split+'_y'];x[ix]=data[split+'_x'][:,-1,exo]
    fill=np.nanmean(x[:cut],axis=0);x=np.where(np.isfinite(x),x,fill)
    _,r,piv=qr(x[:cut]-x[:cut].mean(0),mode='economic',pivoting=True)
    rank=int(np.sum(abs(np.diag(r))>1e-8));cols=sorted(piv[:rank]);x=x[:,cols]
    return x,y,first,cut,[exo[i] for i in cols]


def model(y,x,info,kind,initialization):
    if initialization not in ('approximate_diffuse','exact_diffuse'):raise ValueError('unknown initialization')
    exog=x if x.shape[1] else None
    if kind=='arimax':
        return SARIMAX(y,exog=exog,order=(info['depth'],0,info['depth']),trend='c',enforce_stationarity=False,
                       enforce_invertibility=False,concentrate_scale=False,use_exact_diffuse=initialization=='exact_diffuse')
    if kind=='strx':return UnobservedComponents(y,exog=exog,level='local linear trend',use_exact_diffuse=initialization=='exact_diffuse')
    raise ValueError('repair supports only ARIMAX/STRX')


def replay(data,info,kind,parameters,initialization,qr_basis=False):
    """No fitting or variance estimation: one-step forecasts with frozen parameters."""
    x,y,first,cut,cols=grid(data,info);params=np.asarray(parameters,dtype=float).copy()
    training=model(y[:cut],x[:cut],info,kind,initialization)
    if params.shape!=(len(training.param_names),) or not np.isfinite(params).all():raise ValueError('invalid fixed parameters')
    if any(params[i]<0 for i,name in enumerate(training.param_names) if name.startswith('sigma2')):raise ValueError('negative variance')
    if qr_basis and x.shape[1]:
        _,r=qr(x[:cut],mode='economic');A=np.linalg.solve(r,np.eye(x.shape[1]))*np.sqrt(cut);C=r/np.sqrt(cut)
        beta=[i for i,name in enumerate(training.param_names) if name in training.exog_names or name in ['beta.'+v for v in training.exog_names]]
        params[beta]=C@params[beta];x=x@A;training=model(y[:cut],x[:cut],info,kind,initialization)
    ll=float(training.loglike(params))
    if not np.isfinite(ll):raise ValueError('nonfinite training likelihood')
    combined=model(y,x,info,kind,initialization)
    fitted=combined.filter(params,cov_type='none',low_memory=False)
    all_pred=np.asarray(fitted.predict(start=cut,end=len(y)-1,dynamic=False))
    prediction={split:all_pred[data[split+'_idx']-first-cut] for split in ['val','test']}
    if not all(np.isfinite(p).all() for p in prediction.values()):raise ValueError('nonfinite fixed forecasts')
    return prediction,{'train_loglike':ll,'parameters':np.asarray(parameters,dtype=float).tolist(),'parameter_names':training.param_names,
                       'initialization':initialization,'concentrate_scale':False,'exogenous_original_indices':cols,'first':first,'cut':cut}


def check_candidate(candidate):
    if candidate.get('status')!='converged_training_only_requires_replay':raise ValueError('candidate not converged')
    params=np.asarray(candidate.get('parameters',[]),dtype=float)
    if params.ndim!=1 or not len(params) or not np.isfinite(params).all():raise ValueError('invalid candidate parameters')
    attempts=candidate.get('attempts',[])
    if not attempts:raise ValueError('missing optimizer evidence')
    last=attempts[-1]
    optimizer=last.get('optimizer',{})
    if optimizer.get('converged',False):
        gradient=np.asarray(optimizer.get('gopt',[]),dtype=float)
        if not gradient.size or not np.isfinite(gradient).all() or np.max(abs(gradient))>1e-6:raise ValueError('optimizer score gate failed')
    elif last.get('success') is True:
        score=last.get('max_coordinate_score_per_observation',float('inf'))
        if score is None or not np.isfinite(score) or not 0<=score<=1e-6:raise ValueError('optimizer score gate failed')
        boundary=last.get('max_boundary_variance_violation')
        if boundary is None or not np.isfinite(boundary) or not 0<=boundary<=1e-6:raise ValueError('optimizer boundary gate failed')
    else:raise ValueError('optimizer did not converge')
    directional=candidate.get('final_directional_checks',[])
    if len(directional)!=3:raise ValueError('missing final direction checks')
    for check in directional:
        values=[check.get('score'),check.get('central')]
        if any(v is None for v in values) or not np.isfinite(values).all() or not np.isclose(*values,rtol=.02,atol=2e-6):raise ValueError('final direction score mismatch')
    initial=candidate.get('initial_train_loglike',candidate.get('initial_profile_loglike'))
    final=candidate.get('final_train_loglike',candidate.get('final_profile_loglike'))
    if initial is None or final is None or not np.isfinite([initial,final]).all() or final<initial-1e-6:raise ValueError('training likelihood regressed')
    return params


def replay_checks(data,info,kind,parameters,initialization,expected_ll):
    prediction,details=replay(data,info,kind,parameters,initialization)
    if not np.isclose(details['train_loglike'],expected_ll,rtol=1e-9,atol=1e-6):raise ValueError('training likelihood replay mismatch')
    rotated,rotated_details=replay(data,info,kind,parameters,initialization,qr_basis=True)
    errors={}
    for split in ['val','test']:
        np.testing.assert_allclose(prediction[split],rotated[split],rtol=2e-6,atol=2e-7)
        errors[split]=float(np.max(abs(prediction[split]-rotated[split])))
        for k in sorted(set([0,len(data[split+'_y'])//2,len(data[split+'_y'])-1])):
            changed=dict(data);changed[split+'_y']=data[split+'_y'].copy();changed[split+'_y'][k:]+=1000
            again,detail=replay(changed,info,kind,parameters,initialization)
            np.testing.assert_array_equal(again[split][:k+1],prediction[split][:k+1])
            assert detail['train_loglike']==details['train_loglike']
    return prediction,details,{'status':'PASS','qr_max_absolute_prediction_difference':errors,
                              'causal_checks':'mutated current and all future validation/test targets at first/middle/last rows; prefix forecasts bit-identical',
                              'variance_protocol':'fixed training estimate; concentrate_scale=False during replay'}


def original_evidence(folder):
    return {name:sha(folder/name) for name in ['config.json','result.json','fit_details.json','failure.json','artifacts.json','complete.json'] if (folder/name).exists()}



def check_generation(actual,expected):
    for key in ['data_sha256','data_metadata_sha256','runtime','original_evidence','dependency_sha256']:
        if actual.get(key)!=expected.get(key):raise ValueError('candidate generation provenance mismatch: '+key)


def generation(dataset,kind):
    return {'data_sha256':sha(ROOT/'data/processed'/f'{dataset}.npz'),'data_metadata_sha256':sha(ROOT/'data/processed'/f'{dataset}.json'),
            'runtime':runtime(),'original_evidence':original_evidence(OUT/'classic'/f'{dataset}_{kind}'),'dependency_sha256':sources()}


def check_precondition(precondition,data,info,kind,initialization):
    x,y,first,cut,cols=grid(data,info);training=model(y[:cut],x[:cut],info,kind,initialization)
    if precondition['parameters_original_names']!=training.param_names:raise ValueError('parameter-name mapping mismatch')
    if precondition['first_row']!=first or precondition['cut']!=first+cut or precondition['exogenous_original_indices']!=cols:raise ValueError('training split/design mismatch')
    A=np.asarray(precondition['A']);C=np.asarray(precondition['C'])
    np.testing.assert_allclose(A@C,np.eye(x.shape[1]),rtol=1e-8,atol=1e-8)
    z=x[:cut]@A;np.testing.assert_allclose(z.T@z/cut,np.eye(x.shape[1]),rtol=1e-8,atol=1e-8)


def validate_repair(folder,out=OUT):
    folder=Path(folder)
    if not is_complete(folder):raise ValueError('repair is not complete')
    cfg=json.loads((folder/'config.json').read_text());ds=cfg['dataset'];kind=cfg['model']
    if kind not in ('arimax','strx'):raise ValueError('unsupported repair model')
    if cfg['source_sha256']!=sources() or cfg['runtime']!=runtime():raise ValueError('repair source/runtime changed')
    if cfg['data_sha256']!=sha(ROOT/'data/processed'/f'{ds}.npz') or cfg['data_metadata_sha256']!=sha(ROOT/'data/processed'/f'{ds}.json'):raise ValueError('repair data changed')
    if cfg['original_evidence']!=original_evidence(Path(out)/'classic'/f'{ds}_{kind}'):raise ValueError('original evidence changed')
    if sha(folder/'candidate.json')!=cfg['candidate_sha256'] or sha(folder/'diagnostic_source.py')!=cfg['candidate_source_sha256']:raise ValueError('candidate provenance mismatch')
    candidate=json.loads((folder/'candidate.json').read_text());check_candidate(candidate)
    precondition=json.loads((folder/'precondition.json').read_text())
    expected={'data_sha256':cfg['data_sha256'],'data_metadata_sha256':cfg['data_metadata_sha256'],'runtime':cfg['runtime'],
              'original_evidence':cfg['original_evidence'],'dependency_sha256':cfg['source_sha256']}
    check_generation(candidate['generation_provenance'],expected);check_generation(precondition['generation_provenance'],expected)
    if candidate['generation_provenance']!=precondition['generation_provenance']:raise ValueError('inconsistent generation inputs')
    if json.loads((folder/'attempts.json').read_text())!=candidate['attempts']:raise ValueError('optimizer evidence mismatch')
    for name,digest in cfg['diagnostic_assets_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('diagnostic asset changed')
    for item in cfg['candidate_inputs']:
        if sha(folder/item['archive'])!=item['sha256']:raise ValueError('candidate starting input changed')
    if candidate['source_sha256']!=cfg['candidate_source_sha256']:raise ValueError('candidate source mismatch')
    return cfg,candidate


def resolve_classical(dataset,kind,out=None):
    out=Path(out) if out is not None else OUT;canonical=out/'classic'/f'{dataset}_{kind}'
    registry=json.loads((out/'approved_repairs.json').read_text()) if (out/'approved_repairs.json').exists() else {}
    entry=registry.get(f'{dataset}_{kind}')
    if entry is None:return canonical
    if kind not in ('arimax','strx'):raise ValueError('unsupported repair model')
    folder=(out/entry['folder']).resolve()
    if not folder.is_relative_to((out/'repairs').resolve()):raise ValueError('repair path escapes allowed root')
    if not is_complete(folder):raise ValueError('approved repair is not complete')
    cfg,_=validate_repair(folder,out)
    if cfg['dataset']!=dataset or cfg['model']!=kind:raise ValueError('repair identity mismatch')
    if entry['manifest_sha256']!=sha(folder/'artifacts.json'):raise ValueError('approval manifest mismatch')
    return folder


def verify_repair(folder,out=OUT):
    cfg,candidate=validate_repair(folder,out);data,info=load_data(ROOT/'data/processed'/f"{cfg['dataset']}.npz")
    check_precondition(json.loads((Path(folder)/'precondition.json').read_text()),data,info,cfg['model'],cfg['initialization'])
    expected=candidate['mapped_original_loglike'];p,details,checks=replay_checks(data,info,cfg['model'],candidate['parameters'],cfg['initialization'],expected)
    frame=pd.read_csv(Path(folder)/'predictions_0.csv')
    np.testing.assert_allclose(frame.y_pred_scaled,p['test'],rtol=1e-10,atol=1e-10)
    return {'status':'PASS','run':Path(folder).name,'training_loglike':details['train_loglike'],'checks':checks}



def require_failed_original(folder):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    if is_complete(folder) or result.get('status')=='completed':raise ValueError('successful original cannot be replaced')
    if result.get('status')!='not_converged' and not (folder/'failure.json').exists():raise ValueError('no failed original to repair')


def add_approval(registry,key,entry):
    if key in registry:
        old=registry[key]
        if any(old.get(k)!=entry.get(k) for k in ['folder','manifest_sha256']):raise ValueError('cannot replace existing approval')
        return registry
    return {**registry,key:entry}


def accept(dataset,kind,candidate_dir,source_path,initialization):
    candidate_dir=Path(candidate_dir);source_path=Path(source_path);candidate=json.loads((candidate_dir/'result.json').read_text());parameters=check_candidate(candidate)
    if candidate['dataset']!=dataset or candidate['kind']!=kind:raise ValueError('candidate identity mismatch')
    if candidate['source_sha256']!=sha(source_path):raise ValueError('diagnostic source changed')
    if initialization=='exact_diffuse' and 'exact_diffuse' not in candidate.get('initialization',''):raise ValueError('missing initialization disclosure')
    require_failed_original(OUT/'classic'/f'{dataset}_{kind}')
    expected_generation=generation(dataset,kind);check_generation(candidate['generation_provenance'],expected_generation)
    precondition=json.loads((candidate_dir/'precondition.json').read_text())
    check_generation(precondition['generation_provenance'],expected_generation)
    if precondition['source_sha256']!=sha(source_path):raise ValueError('precondition source mismatch')
    if json.loads((candidate_dir/'attempts.json').read_text())!=candidate['attempts']:raise ValueError('optimizer record mismatch')
    inputs=[]
    for i,(relative,digest) in enumerate(candidate['generation_provenance'].get('input_sha256',{}).items()):
        path=(ROOT.parent/relative).resolve()
        if not path.is_relative_to(ROOT.parent.resolve()) or sha(path)!=digest:raise ValueError('candidate starting input mismatch')
        inputs.append({'original':relative,'archive':f'input_{i}_'+path.name,'sha256':digest})
    cfg={'dataset':dataset,'model':kind,'initialization':initialization,'source_sha256':sources(),'runtime':runtime(),
         'data_sha256':sha(ROOT/'data/processed'/f'{dataset}.npz'),'data_metadata_sha256':sha(ROOT/'data/processed'/f'{dataset}.json'),
         'candidate_sha256':sha(candidate_dir/'result.json'),'candidate_source_sha256':sha(source_path),
         'diagnostic_assets_sha256':{name:sha(candidate_dir/name) for name in ['precondition.json','attempts.json']},'candidate_inputs':inputs,
         'original_evidence':original_evidence(OUT/'classic'/f'{dataset}_{kind}'),'selection_rule':'training-only solver convergence and likelihood; no test-driven selection'}
    name=f"{dataset}_{kind}_{cfg['source_sha256']['repairs.py'][:10]}_{cfg['candidate_sha256'][:10]}";folder=OUT/'repairs'/name
    lock=run_lock(folder,blocking=False);validate_config(folder/'config.json',cfg)
    if not is_complete(folder):
        start=time.monotonic();data,info=load_data(ROOT/'data/processed'/f'{dataset}.npz')
        check_precondition(precondition,data,info,kind,initialization)
        prediction,details,checks=replay_checks(data,info,kind,parameters,initialization,candidate['mapped_original_loglike'])
        shutil.copyfile(candidate_dir/'result.json',folder/'candidate.json');shutil.copyfile(source_path,folder/'diagnostic_source.py')
        for name in ['precondition.json','attempts.json']:shutil.copyfile(candidate_dir/name,folder/name)
        for item in inputs:shutil.copyfile(ROOT.parent/item['original'],folder/item['archive'])
        atomic_json(folder/'fit_details.json',{**details,'converged':True,'repair':True,'optimizer':candidate['attempts'][-1],'checks':checks})
        p=prediction['test'];pd.DataFrame({'target_row':data['test_idx'],'y_true':data['test_raw_y'],'y_pred':p*data['y_scale']+data['y_min'],
                                          'y_true_scaled':data['test_y'],'y_pred_scaled':p}).to_csv(folder/'predictions_0.csv',index=False)
        row={'seed':None,'validation_rmse_scaled':metrics(data['val_y'],prediction['val'])['rmse'],
             'test_scaled':metrics(data['test_y'],p),'test_raw':metrics(data['test_raw_y'],p*data['y_scale']+data['y_min'])}
        atomic_json(folder/'result.json',{'dataset':dataset,'model':kind,'status':'completed','data_sha256':cfg['data_sha256'],'runs':[row],
                                        'seconds':candidate.get('seconds',0)+time.monotonic()-start,'repeat_definition':'deterministic single fit; no artificial repeats',
                                        'repair':True,'initialization':initialization,'original_failure_preserved':True})
        mark_complete(folder)
    proof=verify_repair(folder)
    registry_lock=run_lock(OUT/'repair_registry_lock',blocking=False)
    path=OUT/'approved_repairs.json';registry=json.loads(path.read_text()) if path.exists() else {}
    registry=add_approval(registry,f'{dataset}_{kind}',{'folder':str(folder.relative_to(OUT)),'manifest_sha256':sha(folder/'artifacts.json'),'verification':proof})
    atomic_json(path,registry);resolve_classical(dataset,kind)
    print('ACCEPTED_REPAIR',dataset,kind,folder.name,flush=True)
    return folder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('dataset');parser.add_argument('kind',choices=['arimax','strx']);parser.add_argument('candidate_dir');parser.add_argument('source_path')
    parser.add_argument('--initialization',choices=['approximate_diffuse','exact_diffuse'],required=True);args=parser.parse_args()
    accept(args.dataset,args.kind,args.candidate_dir,args.source_path,args.initialization)

if __name__=='__main__':main()

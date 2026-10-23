"""Validation-only classical baselines, with convergence evidence retained."""
import itertools
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .engine import ROOT,atomic_json,load_data,metrics,sha,file_manifest
from .lifecycle import is_complete,mark_complete,validate_config


def specs(kind):
    if kind=='rf':
        return [{'n_estimators':n,'max_depth':d} for n,d in itertools.product([2,50,100,200],[3,6,10])]
    if kind=='xgt':
        return [{'n_estimators':n,'max_depth':d,'reg_lambda':l} for n,d,l in itertools.product([50,100,200],[3,6,10],[.0001,.001,.01,.1,1.,10.])]
    if kind=='enet':
        return [{'l1':a,'l2':b} for a,b in itertools.product([0.,.1,.3,.5,.7,.9,1.,2.],repeat=2)]
    return [{}]


def estimator(kind,params,seed):
    if kind=='rf': return RandomForestRegressor(**params,random_state=seed,n_jobs=2)
    if kind=='xgt': return XGBRegressor(**params,random_state=seed,n_jobs=2,objective='reg:squarederror',tree_method='hist')
    if kind=='enet':
        total=params['l1']+params['l2']
        if total==0: return LinearRegression(n_jobs=2)
        # sklearn's objective: L1=a*r, L2=(a*(1-r)/2).
        return ElasticNet(alpha=total,l1_ratio=params['l1']/total,max_iter=20000,tol=1e-5,selection='cyclic')
    raise ValueError(kind)


def state_space(d,info,kind):
    """Preserve regular grid with missing y, and never use future exogenous values.

    Exog: last observed non-target variables from the supplied historical window.
    Unknown exog on invalid windows is train-mean filled, its y remains missing.
    Filtering over val/test assimilates only observations from previous steps.
    """
    target=info['target_column']; exo=[i for i in range(len(info['features'])) if i!=target]
    first=int(d['train_idx'][0]); end=int(d['test_idx'][-1])+1
    size=end-first
    y=np.full(size,np.nan); x=np.full((size,len(exo)),np.nan)
    for s in ['train','val','test']:
        ix=d[s+'_idx']-first; y[ix]=d[s+'_y']; x[ix]=d[s+'_x'][:,-1,exo]
    train_end=int(d['val_idx'][0])-first
    fill=np.nanmean(x[:train_end],axis=0)
    x=np.where(np.isfinite(x),x,fill)
    # Remove constant regressors / exact one-hot dummy collinearity using training-only rank.
    import scipy.linalg
    centered=x[:train_end]-x[:train_end].mean(0)
    _,r,piv=scipy.linalg.qr(centered,mode='economic',pivoting=True)
    rank=int(np.sum(np.abs(np.diag(r))>1e-8))
    exo_idx=sorted(piv[:rank]); x=x[:,exo_idx]
    args={'endog':y[:train_end],'exog':x[:train_end] if x.shape[1] else None}
    if kind=='arimax':
        model=SARIMAX(**args,order=(info['depth'],0,info['depth']),trend='c',
                      enforce_stationarity=False,enforce_invertibility=False)
    else:
        model=UnobservedComponents(**args,level='local linear trend')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted=model.fit(disp=False,maxiter=100,low_memory=True)
        fit_attempts=[{'maxiter':100,'converged':bool(fitted.mle_retvals.get('converged',False))}]
        if not fitted.mle_retvals.get('converged',False):
            fitted=model.fit(start_params=fitted.params,disp=False,maxiter=300,low_memory=True)
            fit_attempts.append({'maxiter':300,'converged':bool(fitted.mle_retvals.get('converged',False))})
    all_y=np.concatenate([y[:train_end],y[train_end:]])
    # Fixed parameters. State-space one-step predictions condition on all earlier observations,
    # including validation and earlier test values, never current/future target values.
    if kind=='arimax':
        combined=SARIMAX(all_y,exog=x if x.shape[1] else None,order=(info['depth'],0,info['depth']),trend='c',
                         enforce_stationarity=False,enforce_invertibility=False)
    else:
        combined=UnobservedComponents(all_y,exog=x if x.shape[1] else None,level='local linear trend')
    filtered=combined.filter(fitted.params,low_memory=False)
    prediction=np.asarray(filtered.predict(start=train_end,end=size-1,dynamic=False))
    # Predict with low_memory=False if the installed statespace backend did not retain forecasts.
    result={}
    for s in ['val','test']:
        offsets=d[s+'_idx']-first-train_end
        result[s]=prediction[offsets]
    return result,{'converged':bool(fitted.mle_retvals.get('converged',False)),
                   'optimizer':fitted.mle_retvals,'warnings':[str(w.message) for w in caught],
                   'fit_attempts':fit_attempts,'parameters':np.asarray(fitted.params).tolist(),'exogenous_count':int(x.shape[1]),
                   'exogenous_protocol':'previous observed time only; invalid-window exog=train-mean, y=missing',
                   'order':[info['depth'],0,info['depth']] if kind=='arimax' else 'local linear trend'}


def run_classical(dataset,kind,seeds,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    data_path=ROOT/'data/processed'/f'{dataset}.npz'
    validate_config(folder/'config.json',{'dataset':dataset,'model':kind,'seeds':seeds,'data_sha256':sha(data_path),
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('engine.py'),Path(__file__).with_name('lifecycle.py')]}})
    if is_complete(folder): return json.loads((folder/'result.json').read_text())
    if is_complete(folder,require_converged=False):
        raise RuntimeError('unconverged statistics preserved; review fit_details before further solver changes')
    d,info=load_data(data_path)
    base={'dataset':dataset,'model':kind,'data_sha256':sha(data_path),'status':'completed'}
    start=time.monotonic()
    if kind in ('arimax','strx'):
        predictions,details=state_space(d,info,kind)
        # Preserve a bounded optimizer/convergence record without hiding warnings.
        details['optimizer']={k:(v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v)
                              for k,v in details['optimizer'].items()}
        atomic_json(folder/'fit_details.json',details)
        if not details['converged']: base['status']='not_converged'
        rows=[{'seed':None,'validation_rmse_scaled':metrics(d['val_y'],predictions['val'])['rmse'],
               'test_scaled':metrics(d['test_y'],predictions['test']),
               'test_raw':metrics(d['test_raw_y'],predictions['test']*d['y_scale']+d['y_min'])}]
        test_predictions=[predictions['test']]
    else:
        flat={s:d[s+'_x'].reshape(len(d[s+'_y']),-1) for s in ['train','val','test']}
        search_path=folder/'search.json'; search=json.loads(search_path.read_text()) if search_path.exists() else []
        configurations=specs(kind)
        for i,params in enumerate(configurations):
            if i<len(search): continue
            tic=time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=estimator(kind,params,seeds[0]); model.fit(flat['train'],d['train_y'])
                pred=model.predict(flat['val'])
            score=metrics(d['val_y'],pred)['rmse']
            converged=not any(isinstance(w.message,ConvergenceWarning) for w in caught)
            search.append({'candidate':i,'params':params,'val_rmse_scaled':score,'converged':converged,
                           'warnings':[str(w.message) for w in caught],'seconds':time.monotonic()-tic})
            atomic_json(search_path,search)
            print(f'{dataset}/{kind} candidate {i+1}/{len(configurations)} val={score}',flush=True)
        usable=[s for s in search if s['converged'] and np.isfinite(s['val_rmse_scaled'])]
        if not usable: raise RuntimeError('No converged validation candidate')
        winner=min(usable,key=lambda s:s['val_rmse_scaled'])
        atomic_json(folder/'selected.json',winner)
        actual_seeds=seeds if kind in ('rf','xgt') else [seeds[0]]
        rows=[]; test_predictions=[]
        for seed in actual_seeds:
            model=estimator(kind,winner['params'],seed); model.fit(flat['train'],d['train_y'])
            p=model.predict(flat['test']); test_predictions.append(p)
            rows.append({'seed':seed if kind in ('rf','xgt') else None,
                         'validation_rmse_scaled':winner['val_rmse_scaled'],
                         'test_scaled':metrics(d['test_y'],p),
                         'test_raw':metrics(d['test_raw_y'],p*d['y_scale']+d['y_min'])})
    for j,(p,row) in enumerate(zip(test_predictions,rows)):
        pd.DataFrame({'target_row':d['test_idx'],'y_true':d['test_raw_y'],'y_pred':p*d['y_scale']+d['y_min'],
                      'y_true_scaled':d['test_y'],'y_pred_scaled':p}).to_csv(folder/f'predictions_{j}.csv',index=False)
    result={**base,'runs':rows,'seconds':time.monotonic()-start,
            'repeat_definition':'5 fixed independent random seeds' if kind in ('rf','xgt') else 'deterministic single fit; no artificial repeats'}
    atomic_json(folder/'result.json',result); mark_complete(folder)
    return result

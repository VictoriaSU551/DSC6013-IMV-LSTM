"""Training-only numerical continuation with recorded generation provenance.

QR is invertible. Variances use log coordinates. Initializations and tolerances
are explicit; test metrics never enter starts, objectives or stopping rules.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import uuid
import warnings
import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from .engine import ROOT,sha,atomic_json
from .lifecycle import run_lock
from .repairs import model,generation


def encode(parameters,scale,variance):
    result=np.asarray(parameters,dtype=float)/scale;result=result.copy();result[variance]=np.log(np.asarray(parameters)[variance]);return result


def decode(coordinates,scale,variance):
    result=np.asarray(coordinates)*scale;result=result.copy()
    with np.errstate(over='ignore',under='ignore'):result[variance]=np.exp(np.asarray(coordinates)[variance])
    return result


def chain_score(score,parameters,scale,variance):
    result=np.asarray(score)*scale;result=result.copy();result[variance]=np.asarray(score)[variance]*np.asarray(parameters)[variance];return result


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value



def fresh_attempt(parent,name):
    path=Path(parent)/(name+'_'+uuid.uuid4().hex[:12]);path.mkdir(parents=True,exist_ok=False);return path


def solve(dataset,kind,initialization,warm=None):
    source_sha=sha(Path(__file__));input_hash={}
    if warm is not None:
        warm=Path(warm).resolve()
        if not warm.is_relative_to(ROOT.parent.resolve()):raise ValueError('warm input must be inside isolated project')
        input_hash[str(warm.relative_to(ROOT.parent))]=sha(warm)
    out=fresh_attempt(ROOT.parent/'diagnostics',f'{dataset}_{kind}_{initialization}_attested_{source_sha[:10]}')
    lock=run_lock(out,blocking=False);start=time.monotonic()
    try:
        snapshot=generation(dataset,kind);snapshot['input_sha256']=input_hash
        atomic_json(out/'generation_start.json',{'source_sha256':source_sha,'training_only':True,'generation_provenance':snapshot})
        info=json.loads((ROOT/f'data/processed/{dataset}.json').read_text())
        with np.load(ROOT/f'data/processed/{dataset}.npz') as data:
            tx=data['train_x'];ty=data['train_y'];ix=data['train_idx'];cut=int(data['val_idx'][0])
        exo=[i for i in range(len(info['features'])) if i!=info['target_column']];first=int(ix[0]);n=cut-first
        x=np.full((n,len(exo)),np.nan);y=np.full(n,np.nan);x[ix-first]=tx[:,-1,exo];y[ix-first]=ty
        fill=np.nanmean(x,axis=0);x=np.where(np.isfinite(x),x,fill)
        _,r,piv=qr(x-x.mean(0),mode='economic',pivoting=True);rank=int(np.sum(abs(np.diag(r))>1e-8));columns=sorted(piv[:rank]);x=x[:,columns]
        _,r=qr(x,mode='economic');A=np.linalg.solve(r,np.eye(rank))*np.sqrt(n);C=r/np.sqrt(n);z=x@A
        original=model(y,x,info,kind,initialization);fitted_model=model(y,z,info,kind,initialization)
        names=fitted_model.param_names;exnames=fitted_model.exog_names or []
        beta=[i for i,name in enumerate(names) if name in exnames or name in ['beta.'+v for v in exnames]]
        variance=[i for i,name in enumerate(names) if name.startswith('sigma2')]
        def physical_to_original(p):
            p=np.asarray(p).copy();p[beta]=A@p[beta];return p
        def original_to_physical(p):
            p=np.asarray(p).copy();p[beta]=C@p[beta];return p
        candidates=[]
        def add(label,p):
            p=np.asarray(p,dtype=float)
            if p.shape!=(len(names),) or not np.isfinite(p).all():return
            p=p.copy();p[variance]=np.maximum(p[variance],1e-14)
            try:ll=float(fitted_model.loglike(p))
            except (ValueError,np.linalg.LinAlgError):return
            if np.isfinite(ll):candidates.append((ll,label,p))
        if warm is not None:
            p=np.load(warm) if warm.suffix=='.npy' else np.array(json.loads(warm.read_text())['parameters'])
            add('recorded_training_warm_start',original_to_physical(p))
        good=np.isfinite(y);ols=np.linalg.lstsq(np.column_stack([np.ones(good.sum()),z[good]]),y[good],rcond=None)[0];residual=y-z@ols[1:]
        if kind=='arimax':
            for ar1 in [.5,.8,.95]:
                p=np.zeros(len(names));p[beta]=ols[1:]
                for j in range(1,info['depth']+1):
                    p[names.index('ar.L'+str(j))]=ar1 if j==1 else .005;p[names.index('ma.L'+str(j))]=.1 if j==1 else .005
                phi=np.array([p[names.index('ar.L'+str(j))] for j in range(1,info['depth']+1)])
                p[names.index('intercept')]=(1-phi.sum())*ols[0]
                innovation=residual[info['depth']:].copy()-p[names.index('intercept')]
                for j,coef in enumerate(phi,1):innovation-=coef*residual[info['depth']-j:len(y)-j]
                p[variance]=float(np.nanvar(innovation));add('dense_stable_training_start_'+str(ar1),p)
        else:
            for ratio in [1e-2,1e-4,1e-6]:
                p=np.zeros(len(names));p[beta]=ols[1:];v=float(np.nanvar(np.diff(residual)))
                p[variance]=np.array([.2,.8,ratio])*max(v,1e-8);add('local_trend_training_start_'+str(ratio),p)
        if not candidates:raise RuntimeError('no finite training initialization')
        initial,label,p=max(candidates,key=lambda a:a[0]);ll=float(original.loglike(physical_to_original(p)))
        np.testing.assert_allclose(initial,ll,rtol=1e-9,atol=1e-6)
        scale=np.ones(len(p));scale[beta]=np.sqrt(np.sum(p[variance]));
        if 'intercept' in names:scale[names.index('intercept')]=np.sqrt(np.sum(p[variance]))
        best=[np.inf,None]
        def objective(u):
            v=decode(u,scale,variance)
            if not np.isfinite(v).all() or (v[variance]<=0).any():return np.inf
            try:
                with warnings.catch_warnings():warnings.simplefilter('ignore');value=-float(fitted_model.loglike(v))/n
                if np.isfinite(value) and value<best[0]:best[:]=[value,np.asarray(u).copy()]
                return value if np.isfinite(value) else np.inf
            except (ValueError,np.linalg.LinAlgError):return np.inf
        def gradient(u):
            v=decode(u,scale,variance)
            if not np.isfinite(v).all() or (v[variance]<=0).any():return np.full(len(u),np.nan)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');g=-fitted_model.score(v,transformed=True,method='approx',approx_complex_step=True)/n
                return chain_score(g,v,scale,variance)
            except (ValueError,np.linalg.LinAlgError):return np.full(len(u),np.nan)
        u=encode(p,scale,variance);objective(u);g=gradient(u);directional=[]
        for seed in [11,23,47]:
            direction=np.random.default_rng(seed).normal(size=len(u));direction/=np.linalg.norm(direction);step=1e-5
            analytic=float(g@direction);numeric=float((objective(u+step*direction)-objective(u-step*direction))/(2*step))
            directional.append({'seed':seed,'score':analytic,'central':numeric})
            if not np.isclose(analytic,numeric,rtol=.02,atol=2e-5):raise RuntimeError('independent directional score check failed: '+str(directional[-1]))
        atomic_json(out/'precondition.json',{'source_sha256':source_sha,'training_only':True,'generation_provenance':snapshot,
            'first_row':first,'cut':cut,'exogenous_original_indices':[exo[i] for i in columns],'fill_train':fill.tolist(),
            'A':A.tolist(),'C':C.tolist(),'coordinate_scale':scale.tolist(),'variance_indices':variance,'profile_scale':False,
            'initialization':initialization,'start_label':label,'parameters_original_names':original.param_names,'directional_checks':directional,
            'finite_starts':[{'name':name,'train_loglike':ll} for ll,name,p in candidates]})
        attempts=[];best_ll=initial
        for method,maxiter in [('BFGS',1200),('L-BFGS-B',2000),('Powell',500),('BFGS',1200)]:
            iteration=[0]
            def callback(current):
                iteration[0]+=1
                if iteration[0]%25==0:
                    point=physical_to_original(decode(current,scale,variance))
                    if np.isfinite(point).all():np.save(out/'latest_original_params.npy',point)
                    atomic_json(out/'status.json',{'status':'running','pid':os.getpid(),'method':method,'iteration':iteration[0],'seconds':time.monotonic()-start,'training_only':True})
                    print(method,iteration[0],flush=True)
            atomic_json(out/'status.json',{'status':'running','pid':os.getpid(),'method':method,'seconds':time.monotonic()-start,'training_only':True})
            options={'maxiter':maxiter,'gtol':1e-7}
            if method=='L-BFGS-B':options.update(ftol=1e-14,maxls=40,maxcor=20)
            if method=='Powell':options={'maxiter':maxiter,'xtol':1e-8,'ftol':1e-12}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');fit=minimize(objective,u,jac=None if method=='Powell' else gradient,method=method,callback=callback,options=options)
            point=decode(fit.x,scale,variance);value=-objective(fit.x)*n;gn=float(np.max(abs(gradient(fit.x))))
            boundary=[i for i in variance if point[i]<1e-10];violation=0.
            if boundary and np.isfinite(value):
                physical_score=fitted_model.score(point,transformed=True,method='approx',approx_complex_step=True)/n
                violation=float(max(0,np.max(physical_score[boundary])))
            record={'method':method,'success':bool(fit.success),'message':str(fit.message),'iterations':fit.nit,'train_loglike':value,
                    'max_coordinate_score_per_observation':gn,'max_boundary_variance_violation':violation,'seconds':time.monotonic()-start}
            attempts.append(clean(record));atomic_json(out/'attempts.json',attempts)
            np.save(out/f'{method}_{len(attempts)}_original_params.npy',physical_to_original(point));print(json.dumps(clean(record)),flush=True)
            if best[1] is not None:
                best_ll=-best[0]*n;u=best[1].copy();np.save(out/'best_training_original_params.npy',physical_to_original(decode(u,scale,variance)))
            if fit.success and np.isfinite(value) and value>=initial and value>=best_ll-1e-5 and np.isfinite(gn) and gn<=1e-6 and np.isfinite(violation) and violation<=1e-6 and np.isfinite(point).all():
                final_directions=[];final_gradient=gradient(fit.x)
                for seed in [11,23,47]:
                    direction=np.random.default_rng(seed).normal(size=len(fit.x));direction/=np.linalg.norm(direction);step=1e-5
                    analytic=float(final_gradient@direction);numeric=float((objective(fit.x+step*direction)-objective(fit.x-step*direction))/(2*step))
                    final_directions.append({'seed':seed,'score':analytic,'central':numeric,'step':step})
                    if not np.isclose(analytic,numeric,rtol=.02,atol=2e-6):raise RuntimeError('final independent direction check failed')
                mapped=physical_to_original(point);mapped_ll=float(original.loglike(mapped));np.testing.assert_allclose(value,mapped_ll,rtol=1e-9,atol=1e-6)
                end=generation(dataset,kind);end['input_sha256']={relative:sha(ROOT.parent/relative) for relative in input_hash}
                if snapshot!=end or sha(Path(__file__))!=source_sha:raise RuntimeError('generation inputs/source changed during fitting')
                np.save(out/'converged_original_params.npy',mapped)
                atomic_json(out/'result.json',{'status':'converged_training_only_requires_replay','dataset':dataset,'kind':kind,'source_sha256':source_sha,
                    'generation_provenance':snapshot,'final_directional_checks':final_directions,'initialization':initialization,'parameters':mapped.tolist(),'initial_train_loglike':initial,
                    'final_train_loglike':value,'mapped_original_loglike':mapped_ll,'attempts':attempts,'seconds':time.monotonic()-start})
                atomic_json(out/'status.json',{'status':'converged_training_only_requires_replay','seconds':time.monotonic()-start});return out
        atomic_json(out/'status.json',{'status':'still_not_converged','seconds':time.monotonic()-start});return out
    except Exception as exc:
        atomic_json(out/'status.json',{'status':'diagnostic_failed','error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start});raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('dataset');parser.add_argument('kind',choices=['arimax','strx']);parser.add_argument('--initialization',choices=['exact_diffuse','approximate_diffuse'],required=True);parser.add_argument('--warm');args=parser.parse_args()
    print(solve(args.dataset,args.kind,args.initialization,args.warm),flush=True)

if __name__=='__main__':main()

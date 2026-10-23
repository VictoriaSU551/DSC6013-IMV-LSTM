"""Postfit training-only numerical certification; does not optimize parameters.

A single coarse finite-difference step can have O(h^2) truncation error. Require
three adjacent steps from one fixed grid to agree, keeping the score threshold.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .engine import ROOT,sha,atomic_json,load_data
from . import repairs as v1
from .stats_solver import encode,decode,chain_score,fresh_attempt

STEPS=(1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8,3e-9)


def select_plateau(curve,score):
    if tuple(row.get('step') for row in curve)!=STEPS or not np.isfinite(score):raise ValueError('unexpected finite-difference grid/score')
    for i in range(len(curve)-2):
        window=curve[i:i+3];values=np.array([row['central'] for row in window],dtype=float)
        if np.isfinite(values).all() and np.all(np.isclose(values,score,rtol=.02,atol=2e-6)) and np.ptp(values)<=2e-6:
            chosen=sorted(window,key=lambda row:row['central'])[1]
            return {**chosen,'score':float(score),'plateau':window,'criterion':'first/largest-step three-adjacent agreement; score tolerance unchanged'}
    raise ValueError('no three-adjacent finite-difference plateau')



def collect_parent_inputs(parent,setup,base=None):
    base=(ROOT.parent if base is None else Path(base)).resolve()
    parent=Path(parent).resolve()
    if not parent.is_relative_to(base):raise ValueError('parent outside isolated project')
    started=json.loads((parent/'generation_start.json').read_text())
    if started.get('source_sha256')!=setup.get('source_sha256') or started.get('generation_provenance')!=setup.get('generation_provenance'):
        raise ValueError('optimizer start/provenance mismatch')
    paths=[]
    for relative,digest in setup['generation_provenance'].get('input_sha256',{}).items():
        path=(base/relative).resolve()
        if not path.is_relative_to(base):raise ValueError('optimizer input outside isolated project')
        if sha(path)!=digest:raise ValueError('optimizer input changed')
        paths.append(path)
    return paths


def certify(dataset,kind,parent):
    parent=Path(parent).resolve()
    if not parent.is_relative_to(ROOT.parent.resolve()):raise ValueError('parent outside isolated project')
    setup=json.loads((parent/'precondition.json').read_text());attempts=json.loads((parent/'attempts.json').read_text());last=attempts[-1]
    if last.get('success') is not True or not np.isfinite([last.get('train_loglike',np.nan),last.get('max_coordinate_score_per_observation',np.nan),last.get('max_boundary_variance_violation',np.nan)]).all():raise ValueError('missing successful finite optimizer endpoint')
    if not 0<=last['max_coordinate_score_per_observation']<=1e-6 or not 0<=last['max_boundary_variance_violation']<=1e-6:raise ValueError('optimizer score/boundary failed')
    endpoint=parent/f"{last['method']}_{len(attempts)}_original_params.npy";optimizer_source=ROOT/'code/repro/stats_solver.py'
    if setup['source_sha256']!=sha(optimizer_source) or setup.get('profile_scale') is not False:raise ValueError('unsupported optimizer source/model')
    expected=v1.generation(dataset,kind);v1.check_generation(setup['generation_provenance'],expected)
    parent_inputs=collect_parent_inputs(parent,setup)
    files=list(dict.fromkeys([optimizer_source,parent/'precondition.json',parent/'attempts.json',parent/'generation_start.json',parent/'status.json',endpoint,*parent_inputs]))
    snapshot={**expected,'input_sha256':{str(p.relative_to(ROOT.parent)):sha(p) for p in files}}
    for relative,digest in setup['generation_provenance'].get('input_sha256',{}).items():
        normalized=str((ROOT.parent/relative).resolve().relative_to(ROOT.parent))
        if snapshot['input_sha256'][normalized]!=digest:raise ValueError('optimizer input changed during snapshot')
    source=sha(Path(__file__));out=fresh_attempt(ROOT.parent/'diagnostics',f'{dataset}_{kind}_postfit_certificate_{source[:10]}');lock=v1.run_lock(out,blocking=False);start=time.monotonic()
    atomic_json(out/'generation_start.json',{'source_sha256':source,'phase':'postfit numerical certification; no parameter optimization','generation_provenance':snapshot})
    parameters=np.load(endpoint);data,info=load_data(ROOT/'data/processed'/f'{dataset}.npz');x,y,first,cut,cols=v1.grid(data,info);x=x[:cut];y=y[:cut]
    initialization=setup['initialization'];original=v1.model(y,x,info,kind,initialization);mapped_ll=float(original.loglike(parameters))
    p=parameters.copy();exnames=original.exog_names or [];beta=[i for i,n in enumerate(original.param_names) if n in exnames or n in ['beta.'+a for a in exnames]]
    p[beta]=np.array(setup['C'])@p[beta];model=v1.model(y,x@np.array(setup['A']),info,kind,initialization)
    scale=np.array(setup['coordinate_scale']);variance=setup['variance_indices'];u=encode(p,scale,variance)
    def objective(value):return -model.loglike(decode(value,scale,variance))/cut
    physical=model.score(p,transformed=True,method='approx',approx_complex_step=True)/cut;gradient=-chain_score(physical,p,scale,variance)
    norm=float(np.max(abs(gradient)));boundary=[i for i in variance if p[i]<1e-10];violation=float(max(0,np.max(physical[boundary]))) if boundary else 0.
    if not np.isfinite(norm) or norm>1e-6 or not np.isfinite(violation) or violation>1e-6:raise ValueError('recomputed score/boundary failed')
    ll=-float(objective(u))*cut;np.testing.assert_allclose(ll,last['train_loglike'],rtol=1e-9,atol=1e-6);np.testing.assert_allclose(mapped_ll,ll,rtol=1e-9,atol=1e-6)
    checks=[];curves=[]
    for seed in [11,23,47]:
        direction=np.random.default_rng(seed).normal(size=len(u));direction/=np.linalg.norm(direction);score=float(gradient@direction);curve=[]
        for step in STEPS:curve.append({'step':step,'central':float((objective(u+step*direction)-objective(u-step*direction))/(2*step))})
        checks.append({'seed':seed,**select_plateau(curve,score)});curves.append({'seed':seed,'score':score,'curve':curve})
    end=v1.generation(dataset,kind);end['input_sha256']={str(p.relative_to(ROOT.parent)):sha(p) for p in files}
    if end!=snapshot or sha(Path(__file__))!=source:raise ValueError('certification source/inputs changed')
    precondition={**setup,'source_sha256':source,'generation_provenance':snapshot,'certification_only':True,
                  'optimizer_parent_source_sha256':setup['source_sha256'],'optimizer_parent_generation_provenance':setup['generation_provenance']}
    atomic_json(out/'precondition.json',precondition);shutil.copyfile(parent/'attempts.json',out/'attempts.json')
    initial=max(row['train_loglike'] for row in setup['finite_starts'])
    result={'status':'converged_training_only_requires_replay','dataset':dataset,'kind':kind,'source_sha256':source,
        'generation_provenance':snapshot,'initialization':initialization,'parameters':parameters.tolist(),'initial_train_loglike':initial,
        'final_train_loglike':ll,'mapped_original_loglike':mapped_ll,'attempts':attempts,'final_directional_checks':checks,'final_multistep_curves':curves,
        'optimization_performed':False,'phase':'postfit numerical certification of preserved optimizer endpoint','optimizer_parent':str(parent),
        'recomputed_max_coordinate_score_per_observation':norm,'recomputed_boundary_variance_violation':violation,
        'seconds':last['seconds']+time.monotonic()-start,'certification_seconds':time.monotonic()-start}
    v1.check_candidate(result);atomic_json(out/'result.json',result);atomic_json(out/'status.json',{'status':result['status'],'certification_only':True,'seconds':time.monotonic()-start})
    print('CERTIFIED_ENDPOINT',out,flush=True);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('kind',choices=['strx','arimax']);p.add_argument('parent');a=p.parse_args();certify(a.dataset,a.kind,a.parent)

if __name__=='__main__':main()

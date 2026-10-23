"""Training-only common-coordinate audit; raw optimizer gopt is not comparable."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from repro.engine import ROOT,load_data,atomic_json,sha
from repro.repairs import resolve_classical,grid,model,sources,runtime
from repro.stats_solver import encode,decode,chain_score
out={};base=ROOT/'output/reproduction'
metadata={ds:sha(ROOT/f'data/processed/{ds}.json') for ds in ['sml','pm25','nasdaq']}
data_hash={ds:sha(ROOT/f'data/processed/{ds}.npz') for ds in ['sml','pm25','nasdaq']}
initial_dependencies=sources();initial_runtime=runtime()
for ds in ['sml','pm25','nasdaq']:
 data,info=load_data(ROOT/'data/processed'/f'{ds}.npz')
 for kind in ['strx','arimax']:
  folder=resolve_classical(ds,kind);name=f'{ds}_{kind}'
  if not (folder/'fit_details.json').exists():out[name]={'status':'pending'};continue
  detail=json.loads((folder/'fit_details.json').read_text());p=np.array(detail['parameters']);init=detail.get('initialization','approximate_diffuse')
  x,y,first,cut,cols=grid(data,info);x=x[:cut];y=y[:cut];m=model(y,x,info,kind,init)
  beta=[i for i,n in enumerate(m.param_names) if n in m.exog_names or n in ['beta.'+a for a in m.exog_names]];vi=[i for i,n in enumerate(m.param_names) if n.startswith('sigma2')]
  _,r=qr(x,mode='economic');A=np.linalg.solve(r,np.eye(x.shape[1]))*np.sqrt(cut);C=r/np.sqrt(cut);p[beta]=C@p[beta];x=x@A;m=model(y,x,info,kind,init)
  original_variances=p[vi].copy();p[vi]=np.maximum(p[vi],1e-14);scale=np.ones(len(p));scale[beta]=np.sqrt(np.sum(p[vi]))
  if 'intercept' in m.param_names:scale[m.param_names.index('intercept')]=np.sqrt(np.sum(p[vi]))
  def objective(u):return -m.loglike(decode(u,scale,vi))/cut
  u=encode(p,scale,vi);physical=m.score(p,transformed=True,method='approx',approx_complex_step=True)/cut;g=-chain_score(physical,p,scale,vi)
  boundary=[i for i in vi if p[i]<1e-10];violation=float(max(0,np.max(physical[boundary]))) if boundary else 0.
  checks=[]
  for seed in [11,23,47]:
   v=np.random.default_rng(seed).normal(size=len(u));v/=np.linalg.norm(v);a=float(g@v);b=float((objective(u+v*1e-5)-objective(u-v*1e-5))/(2e-5));checks.append({'score':a,'central':b,'pass':bool(np.isclose(a,b,rtol=.02,atol=2e-6))})
  gn=float(np.max(abs(g)));passed=bool(np.isfinite(gn) and gn<=1e-6 and np.isfinite(violation) and violation<=1e-6 and all(r['pass'] for r in checks))
  out[name]={'status':'PASS' if passed else 'needs_training_diagnosis','score':gn,'boundary_violation':violation,'directions':checks,'initialization':init,'original_variances':original_variances.tolist(),'folder':str(folder.relative_to(base)),
             'result_sha256':sha(folder/'result.json'),'manifest_sha256':sha(folder/'artifacts.json'),'parameters_sha256':sha(folder/'fit_details.json'),'data_sha256':sha(ROOT/f'data/processed/{ds}.npz')}
  print(name,out[name]['status'],gn,violation,checks,flush=True)
assert metadata=={ds:sha(ROOT/f'data/processed/{ds}.json') for ds in metadata}
assert data_hash=={ds:sha(ROOT/f'data/processed/{ds}.npz') for ds in data_hash}
assert initial_dependencies==sources() and initial_runtime==runtime()
atomic_json(base/'statistical_common_coordinate_audit.json',{'status':'PASS' if all(r['status']=='PASS' for r in out.values()) else 'pending_or_diagnosis','rule':'same QR, log variance, regression SD scale; score<=1e-6, boundary<=1e-6, three central directions atol2e-6','source_sha256':sha(Path(__file__)),'dependencies':initial_dependencies,'runtime':initial_runtime,'data_metadata_sha256':metadata,'data_sha256':data_hash,'training_only':True,'groups':out})

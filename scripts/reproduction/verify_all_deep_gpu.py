"""Independent deep audit with source- and artifact-bound resumable evidence."""
import json,time,fcntl,os,inspect,hashlib,traceback,subprocess
from pathlib import Path
import torch
from repro.verify import expected_runs,verify_run,OUT
from repro.engine import ROOT,sha,atomic_json
from repro.lifecycle import is_complete
b=ROOT.parent;lock=(b/'deep_verify.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2)
assert torch.cuda.is_available()
torch.cuda.set_per_process_memory_fraction(.60)
state={'function_sha256':hashlib.sha256(inspect.getsource(verify_run).encode()).hexdigest(),
       'source_sha256':{name:sha(ROOT/'code/repro'/name) for name in ['models.py','engine.py','lifecycle.py']},
       'data_sha256':{p.name:sha(p) for p in sorted((ROOT/'data/processed').glob('*')) if p.suffix in ('.npz','.json')}}
identifier=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()[:12];path=OUT/f'deep_audit_{identifier}.json'
prior=json.loads(path.read_text()) if path.exists() else {}
if prior:
 for key,want in state.items():assert prior.get(key)==want,('audit identity changed',key)
accepted={r['run']:r for r in prior.get('runs',[])};results=[];start=time.monotonic()
def proof(folder):return {name:sha(folder/name) for name in ['config.json','result.json','artifacts.json','complete.json']}
try:
 for folder in expected_runs():
  p=proof(folder);old=accepted.get(folder.name)
  if old and old.get('proof')==p and is_complete(folder):row=old
  else:
   torch.cuda.empty_cache()
   while True:
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    if not any(pid!=str(os.getpid()) for pid in pids):break
    atomic_json(b/'deep_verification_status.json',{'status':'waiting_for_free_gpu','pid':os.getpid(),'verified':len(results),'expected':195,'audit':str(path)})
    time.sleep(60)
   row={**verify_run(folder,device='cuda'),'proof':p,'verification_device':'cuda'}
  results.append(row)
  atomic_json(path,{**state,'status':'running','verified':len(results),'expected':195,'runs':results})
  atomic_json(b/'deep_verification_status.json',{'status':'running','pid':os.getpid(),'verified':len(results),'expected':195,'current':folder.name,'seconds':time.monotonic()-start,'audit':str(path)})
  print(folder.name,'PASS',flush=True)
 assert len(results)==195
 for name,digest in state['source_sha256'].items():assert sha(ROOT/'code/repro'/name)==digest
 for name,digest in state['data_sha256'].items():assert sha(ROOT/'data/processed'/name)==digest
 atomic_json(path,{**state,'status':'PASS','verified':len(results),'expected':195,'runs':results})
 atomic_json(b/'deep_verification_status.json',{'status':'PASS','verified':195,'expected':195,'seconds':time.monotonic()-start,'audit':str(path)})
except Exception as exc:
 atomic_json(b/'deep_verification_status.json',{'status':'failed','verified':len(results),'current':folder.name,'error':str(exc),'traceback':traceback.format_exc(),'audit':str(path)});raise

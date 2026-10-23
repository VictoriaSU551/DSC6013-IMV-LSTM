"""Supervise approved training-only continuations; never replace evidence."""
import os,json,time,fcntl,subprocess,sys,traceback
from pathlib import Path
from repro.engine import ROOT,atomic_json,sha
from repro.repairs import accept
from repro.audited_repairs import accept_audited,resolve_classical,is_strict_complete
from repro.lifecycle import is_complete
b=ROOT.parent;out=ROOT/'output/reproduction';lock=(b/'suite.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
patterns={key:key+'_exact_diffuse_attested_6d71fea26b_*' for key in ['pm25_arimax','sml_strx','nasdaq_strx','nasdaq_arimax']}
audit=out/'statistical_audit_certificate_d05027a4c3b9.json';bad=['sml_strx','nasdaq_strx','nasdaq_arimax'];errors={}
atomic_json(b/'strict_recovery_tasks.json',{'coordinator_pid':os.getpid(),'script':str(Path(__file__)),'patterns':patterns,'audit_certificate':str(audit),
 'next_step':'v2 reviewed and enabled; automatic exact-certificate approval, final strict audit, verified report draft; agent review/publish/delivery remain','log':str(b/'logs/strict_followup_watch.log')})
atomic_json(out/'active_worker.json',{'pid':os.getpid(),'category':'strict_recovery','name':'statistical_recovery','log':str(b/'logs/strict_followup_watch.log')})
try:
 while True:
  states={};live=False
  for key,pattern in patterns.items():
   folders=list((b/'diagnostics').glob(pattern));assert len(folders)<=1,(key,'ambiguous attempt')
   if not folders:states[key]={'status':'not_started'};continue
   folder=folders[0];p=folder/'status.json';s=json.loads(p.read_text()) if p.exists() else {'status':'initializing'}
   pid=s.get('pid');alive=bool(pid and Path('/proc/'+str(pid)).exists());live=live or alive
   states[key]={**{k:s.get(k) for k in ['status','pid','method','iteration','seconds','error']},'alive':alive,'folder':str(folder)}
   if key in bad and s.get('status')=='converged_training_only_requires_replay' and key not in errors:
    registry=json.loads((out/'approved_repairs.json').read_text()) if (out/'approved_repairs.json').exists() else {}
    if key not in registry:
     ds,kind=key.rsplit('_',1)
     try:accept_audited(ds,kind,folder,ROOT/'code/repro/stats_solver.py',audit,ROOT/'code/repro/audit_stats.py','exact_diffuse')
     except Exception as exc:errors[key]=str(exc);print('AUDITED_ACCEPTANCE_ERROR',key,str(exc),flush=True)
   if key=='pm25_arimax' and s.get('status')=='converged_training_only_requires_replay' and key not in errors:
    registry=json.loads((out/'approved_repairs.json').read_text()) if (out/'approved_repairs.json').exists() else {}
    if key not in registry:
     try:accept('pm25','arimax',folder,ROOT/'code/repro/stats_solver.py','exact_diffuse')
     except Exception as exc:errors[key]=str(exc);print('ACCEPTANCE_ERROR',key,str(exc),flush=True)
  raw_done=0;strict_done=0;pending=[]
  for ds in ['sml','pm25','nasdaq']:
   for kind in ['enet','rf','xgt','strx','arimax']:
    key=f'{ds}_{kind}';folder=resolve_classical(ds,kind);ok=is_complete(folder);raw_done+=int(ok)
    strict_ok=is_strict_complete(ds,kind)
    strict_done+=int(strict_ok)
    if not strict_ok:pending.append(key)
  deep=json.loads((b/'deep_verification_status.json').read_text());deep_running=deep.get('status') in ['running','waiting_for_free_gpu']
  state={'stage':'strict_statistical_recovery','status':'running' if live or deep_running else 'awaiting_strict_approval',
         'completed_units':347+strict_done,'total_units':363,'percent':round((347+strict_done)/363*100,2),'hpo':150,'final':105,'selection':90,
         'classical_groups':strict_done,'raw_resolved_classical_groups':raw_done,'strict_audit_pending':pending,'coordinator_pid':os.getpid(),
         'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'count_adjustment':'three original optimizer flags fail additional common-coordinate training audit; old results preserved'}
  atomic_json(b/'progress.json',state);atomic_json(b/'strict_recovery_status.json',{'status':state['status'],'tasks':states,'acceptance_errors':errors,'deep_verified':deep.get('verified'),'strict_pending':pending})
  if strict_done==15 and deep.get('status')=='PASS':
   subprocess.run([sys.executable,'-B','-m','repro.audit_stats_multistep'],cwd=ROOT,env=os.environ.copy(),check=True)
   final_audit=json.loads((out/'statistical_multistep_audit.json').read_text())
   if final_audit['status']!='PASS' or len(final_audit['groups'])!=6:raise RuntimeError('final common-coordinate training audit did not pass')
   atomic_json(out/'strict_final_coordinate_audit.json',final_audit)
   env={**os.environ,'REPRO_VERIFY_DEVICE':'cpu'}
   subprocess.run([sys.executable,'-B','-m','repro.verify','--deep-audit',deep['audit']],cwd=ROOT,env=env,check=True)
   subprocess.run([sys.executable,'-B','-m','repro.report'],cwd=ROOT,env=env,check=True)
   atomic_json(out/'delivery_ready.json',{'status':'awaiting_agent_review_and_publish','report':str(b/'delivery')})
   state.update(stage='awaiting_agent_review_and_publish',status='awaiting_agent_review_and_publish',completed_units=363,percent=100.0);atomic_json(b/'progress.json',state);break
  if not live and not deep_running:break
  time.sleep(60)
except Exception as exc:
 atomic_json(b/'strict_recovery_failure.json',{'error':str(exc),'traceback':traceback.format_exc()});raise

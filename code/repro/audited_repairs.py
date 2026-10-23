"""Versioned approval extension for explicitly rejected training audits.

The v1 training/replay implementation stays immutable. Completed originals may
only receive a sidecar when the archived common-coordinate audit rejects that
specific result. Existing approvals cannot be replaced.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from . import repairs as v1
from .engine import ROOT,sha,atomic_json,load_data,metrics
from .lifecycle import is_complete,mark_complete,run_lock,validate_config

OUT=ROOT/'output/reproduction'
POLICY_ROOT_SHA='b8c0b211b7d290959923e5837534b465526bfc2f63f6ca12f41c3f4d0d69aba9'
AUDIT_RULE='same QR, log variance, regression SD scale; score<=1e-6, boundary<=1e-6, three central directions atol2e-6'


def policy_entries():
    path=ROOT/'configs/statistical_audit_policy.json'
    if sha(path)!=POLICY_ROOT_SHA:raise ValueError('untrusted audit policy root')
    policy=json.loads(path.read_text())
    if policy.get('version')!=2 or not isinstance(policy.get('certificates'),list):raise ValueError('unsupported audit policy')
    return policy['certificates']


def validate_certificate(certificate_path,dataset,kind,program_sha):
    certificate_path=Path(certificate_path).resolve()
    if not certificate_path.is_relative_to(ROOT.parent.resolve()):raise ValueError('audit certificate outside isolated project')
    certificate_sha=sha(certificate_path);key=f'{dataset}_{kind}'
    trusted=[entry for entry in policy_entries() if entry.get('sha256')==certificate_sha and entry.get('program_sha256')==program_sha and key in entry.get('rejected_groups',[])]
    if len(trusted)!=1:raise ValueError('untrusted audit certificate/program')
    certificate=json.loads(certificate_path.read_text())
    if certificate.get('training_only') is not True or certificate.get('rule')!=AUDIT_RULE:raise ValueError('unsupported audit rule')
    if certificate.get('source_sha256')!=program_sha:raise ValueError('audit program mismatch')
    if certificate.get('dependencies')!=v1.sources() or certificate.get('runtime')!=v1.runtime():raise ValueError('audit source/runtime mismatch')
    if kind not in ('strx','arimax') or dataset not in ('sml','pm25','nasdaq'):raise ValueError('unsupported audit group')
    rows=certificate.get('groups',{});key=f'{dataset}_{kind}'
    if not isinstance(rows,dict) or key not in rows:raise ValueError('audit group missing')
    row=rows[key]
    if row.get('status')!='needs_training_diagnosis' or row.get('folder')!=f'classic/{key}':raise ValueError('audit does not reject this original')
    score=row.get('score');boundary=row.get('boundary_violation');directions=row.get('directions',[])
    if score is None or boundary is None or not np.isfinite([score,boundary]).all() or min(score,boundary)<0:raise ValueError('invalid numerical audit')
    if len(directions)!=3 or any(not np.isfinite([c.get('score',np.nan),c.get('central',np.nan)]).all() for c in directions):raise ValueError('invalid audit directions')
    if score<=1e-6 and boundary<=1e-6 and all(np.isclose(c['score'],c['central'],rtol=.02,atol=2e-6) for c in directions):raise ValueError('audit has no numerical rejection')
    data_sha=sha(ROOT/'data/processed'/f'{dataset}.npz');meta_sha=sha(ROOT/'data/processed'/f'{dataset}.json')
    if not isinstance(certificate.get('data_sha256'),dict) or certificate['data_sha256'].get(dataset)!=data_sha or row.get('data_sha256')!=data_sha:raise ValueError('audit data mismatch')
    if not isinstance(certificate.get('data_metadata_sha256'),dict) or certificate['data_metadata_sha256'].get(dataset)!=meta_sha:raise ValueError('audit metadata mismatch')
    folder=OUT/'classic'/key
    if not is_complete(folder) or json.loads((folder/'result.json').read_text()).get('status')!='completed':raise ValueError('audit extension only supports preserved completed originals')
    for filename,field in [('result.json','result_sha256'),('artifacts.json','manifest_sha256'),('fit_details.json','parameters_sha256')]:
        if sha(folder/filename)!=row.get(field):raise ValueError('audited original identity mismatch')
    return row


def validate_policy(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
    version=cfg.get('approval_policy_version',1)
    if type(version) is not int or version not in (1,2):raise ValueError('unsupported approval policy version')
    if version==1:return cfg
    policy_sha=cfg['approval_policy_source_sha256']
    if policy_sha!=sha(Path(__file__)) or sha(folder/'approval_policy.py')!=policy_sha:raise ValueError('approval policy changed')
    if sha(folder/'audit_certificate.json')!=cfg['audit_certificate_sha256'] or sha(folder/'audit_program.py')!=cfg['audit_program_sha256']:raise ValueError('audit certificate/program changed')
    if sha(folder/'audit_policy.json')!=cfg['audit_policy_root_sha256'] or cfg['audit_policy_root_sha256']!=POLICY_ROOT_SHA:raise ValueError('audit policy trust root changed')
    validate_certificate(folder/'audit_certificate.json',cfg['dataset'],cfg['model'],cfg['audit_program_sha256'])
    return cfg


def resolve_classical(dataset,kind,out=None):
    folder=v1.resolve_classical(dataset,kind,out)
    if folder.parent.name=='repairs':validate_policy(folder)
    return folder



def is_strict_complete(dataset,kind,out=None):
    folder=resolve_classical(dataset,kind,out)
    if not is_complete(folder):return False
    key=f'{dataset}_{kind}';rejections=[entry for entry in policy_entries() if key in entry.get('rejected_groups',[])]
    if rejections:
        if folder.parent.name=='classic':
            for entry in rejections:
                path=(ROOT/entry['path']).resolve()
                if not path.is_relative_to(OUT.resolve()):raise ValueError('pinned audit path outside output')
                validate_certificate(path,dataset,kind,entry['program_sha256'])
            return False
        cfg=json.loads((folder/'config.json').read_text())
        if cfg.get('approval_policy_version')!=2:raise ValueError('audit-rejected group requires v2 approval')
    return True


def require_strict_complete(dataset,kind,out=None):
    if not is_strict_complete(dataset,kind,out):raise RuntimeError('strict statistical audit unresolved: '+dataset+'_'+kind)
    return resolve_classical(dataset,kind,out)


def verify_repair(folder,out=None):
    validate_policy(folder)
    return v1.verify_repair(folder,v1.OUT if out is None else out)


def accept_audited(dataset,kind,candidate_dir,source_path,certificate_path,program_path,initialization):
    candidate_dir=Path(candidate_dir);source_path=Path(source_path);certificate_path=Path(certificate_path);program_path=Path(program_path)
    if any(not path.resolve().is_relative_to(ROOT.parent.resolve()) for path in [candidate_dir,source_path,certificate_path,program_path]):raise ValueError('approval input outside isolated project')
    validate_certificate(certificate_path,dataset,kind,sha(program_path))
    candidate=json.loads((candidate_dir/'result.json').read_text());parameters=v1.check_candidate(candidate)
    if candidate.get('dataset')!=dataset or candidate.get('kind')!=kind or candidate.get('initialization')!=initialization:raise ValueError('candidate identity/initialization mismatch')
    if candidate.get('source_sha256')!=sha(source_path):raise ValueError('candidate source changed')
    expected=v1.generation(dataset,kind);precondition=json.loads((candidate_dir/'precondition.json').read_text())
    v1.check_generation(candidate['generation_provenance'],expected);v1.check_generation(precondition['generation_provenance'],expected)
    if candidate['generation_provenance']!=precondition['generation_provenance'] or precondition['source_sha256']!=sha(source_path):raise ValueError('inconsistent candidate generation evidence')
    if json.loads((candidate_dir/'attempts.json').read_text())!=candidate['attempts']:raise ValueError('candidate optimizer evidence mismatch')
    inputs=[]
    for i,(relative,digest) in enumerate(candidate['generation_provenance'].get('input_sha256',{}).items()):
        path=(ROOT.parent/relative).resolve()
        if not path.is_relative_to(ROOT.parent.resolve()) or sha(path)!=digest:raise ValueError('candidate starting input mismatch')
        inputs.append({'original':relative,'archive':f'input_{i}_'+path.name,'sha256':digest})
    cfg={'dataset':dataset,'model':kind,'initialization':initialization,'source_sha256':v1.sources(),'runtime':v1.runtime(),
         'data_sha256':expected['data_sha256'],'data_metadata_sha256':expected['data_metadata_sha256'],'original_evidence':expected['original_evidence'],
         'candidate_sha256':sha(candidate_dir/'result.json'),'candidate_source_sha256':sha(source_path),
         'diagnostic_assets_sha256':{name:sha(candidate_dir/name) for name in ['precondition.json','attempts.json']},'candidate_inputs':inputs,
         'selection_rule':'training-only strict numerical audit and convergence; no test-driven selection',
         'approval_policy_version':2,'approval_policy_source_sha256':sha(Path(__file__)),
         'audit_certificate_sha256':sha(certificate_path),'audit_program_sha256':sha(program_path),'audit_policy_root_sha256':POLICY_ROOT_SHA}
    name=f"{dataset}_{kind}_auditv2_{cfg['approval_policy_source_sha256'][:8]}_{cfg['candidate_sha256'][:8]}_{cfg['audit_certificate_sha256'][:8]}";folder=OUT/'repairs'/name
    key=f'{dataset}_{kind}';path=OUT/'approved_repairs.json';existing=json.loads(path.read_text()) if path.exists() else {}
    if key in existing and existing[key]['folder']!=str(folder.relative_to(OUT)):raise ValueError('cannot replace existing approval')
    lock=run_lock(folder,blocking=False);validate_config(folder/'config.json',cfg)
    if not is_complete(folder):
        start=time.monotonic();data,info=load_data(ROOT/'data/processed'/f'{dataset}.npz');v1.check_precondition(precondition,data,info,kind,initialization)
        prediction,detail,checks=v1.replay_checks(data,info,kind,parameters,initialization,candidate['mapped_original_loglike'])
        for original,archive in [(candidate_dir/'result.json','candidate.json'),(source_path,'diagnostic_source.py'),
                                  (candidate_dir/'precondition.json','precondition.json'),(candidate_dir/'attempts.json','attempts.json'),
                                  (certificate_path,'audit_certificate.json'),(program_path,'audit_program.py'),(Path(__file__),'approval_policy.py'),
                                  (ROOT/'configs/statistical_audit_policy.json','audit_policy.json')]:shutil.copyfile(original,folder/archive)
        for item in inputs:shutil.copyfile(ROOT.parent/item['original'],folder/item['archive'])
        atomic_json(folder/'fit_details.json',{**detail,'converged':True,'repair':True,'optimizer':candidate['attempts'][-1],'checks':checks,'approval_policy_version':2})
        p=prediction['test'];pd.DataFrame({'target_row':data['test_idx'],'y_true':data['test_raw_y'],'y_pred':p*data['y_scale']+data['y_min'],
                                         'y_true_scaled':data['test_y'],'y_pred_scaled':p}).to_csv(folder/'predictions_0.csv',index=False)
        row={'seed':None,'validation_rmse_scaled':metrics(data['val_y'],prediction['val'])['rmse'],'test_scaled':metrics(data['test_y'],p),
             'test_raw':metrics(data['test_raw_y'],p*data['y_scale']+data['y_min'])}
        atomic_json(folder/'result.json',{'dataset':dataset,'model':kind,'status':'completed','data_sha256':cfg['data_sha256'],'runs':[row],
            'seconds':candidate['seconds']+time.monotonic()-start,'repeat_definition':'deterministic single fit; no artificial repeats','repair':True,
            'initialization':initialization,'approval_policy_version':2,'original_audit_rejection_preserved':True})
        mark_complete(folder)
    proof=verify_repair(folder);registry_lock=run_lock(OUT/'repair_registry_lock',blocking=False)
    registry=json.loads(path.read_text()) if path.exists() else {};registry=v1.add_approval(registry,key,{'folder':str(folder.relative_to(OUT)),
        'manifest_sha256':sha(folder/'artifacts.json'),'verification':proof,'approval_policy_version':2})
    atomic_json(path,registry);resolve_classical(dataset,kind);print('ACCEPTED_AUDITED_REPAIR',key,folder.name,flush=True);return folder


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('kind',choices=['strx','arimax']);p.add_argument('candidate_dir');p.add_argument('source_path');p.add_argument('certificate_path');p.add_argument('program_path')
    p.add_argument('--initialization',choices=['exact_diffuse','approximate_diffuse'],required=True);a=p.parse_args()
    accept_audited(a.dataset,a.kind,a.candidate_dir,a.source_path,a.certificate_path,a.program_path,a.initialization)

if __name__=='__main__':main()

import copy
import pytest
from repro.verify import check_audit_header


def audit_fixture():
    return {'status':'PASS','verified':2,'expected':2,'function_sha256':'f','source_sha256':{'models.py':'s'},'data_sha256':{'data.npz':'d'},
            'runs':[{'run':'a','status':'PASS'},{'run':'b','status':'PASS'}]}


def test_deep_replay_evidence_requires_exact_sources_data_and_complete_run_set():
    audit=audit_fixture();expected={'function_sha256':'f','source_sha256':{'models.py':'s'},'data_sha256':{'data.npz':'d'}}
    assert set(check_audit_header(audit,expected,['a','b']))=={'a','b'}
    for key in ['status','function_sha256','source_sha256','data_sha256','verified']:
        changed=copy.deepcopy(audit);changed[key]='changed'
        with pytest.raises(ValueError):check_audit_header(changed,expected,['a','b'])
    for rows in [[audit['runs'][0]],[audit['runs'][0],audit['runs'][0]]]:
        changed=copy.deepcopy(audit);changed['runs']=rows
        with pytest.raises(ValueError):check_audit_header(changed,expected,['a','b'])


def test_reusing_audit_still_rejects_changed_checkpoint(tmp_path,monkeypatch):
    import hashlib,inspect,json
    import repro.verify as verifier
    from repro.lifecycle import write_json,mark_complete,digest
    project=tmp_path/'project';out=project/'output';(project/'code/repro').mkdir(parents=True);(project/'data/processed').mkdir(parents=True)
    for name in ['models.py','engine.py','lifecycle.py']:(project/'code/repro'/name).write_text(name)
    (project/'data/processed/example.npz').write_bytes(b'data')
    monkeypatch.setattr(verifier,'ROOT',project);monkeypatch.setattr(verifier,'OUT',out)
    folders=[out/'run_a',out/'run_b'];rows=[]
    for folder in folders:
        write_json(folder/'config.json',{});write_json(folder/'result.json',{'status':'completed'});(folder/'best.pt').write_bytes(b'original');mark_complete(folder)
        rows.append({'run':folder.name,'status':'PASS','proof':{name:digest(folder/name) for name in ['config.json','result.json','artifacts.json','complete.json']}})
    audit={'status':'PASS','verified':2,'expected':2,'function_sha256':hashlib.sha256(inspect.getsource(verifier.verify_run).encode()).hexdigest(),
           'source_sha256':{name:digest(project/'code/repro'/name) for name in ['models.py','engine.py','lifecycle.py']},
           'data_sha256':{'example.npz':digest(project/'data/processed/example.npz')},'runs':rows}
    path=out/'audit.json';write_json(path,audit);assert len(verifier.load_deep_audit(path,folders))==2
    (folders[0]/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):verifier.load_deep_audit(path,folders)

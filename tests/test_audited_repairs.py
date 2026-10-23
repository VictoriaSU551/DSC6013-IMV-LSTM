"""Only a specific immutable failed training audit authorizes a completed refit."""
import copy
import json
from pathlib import Path
import pytest
from repro import audited_repairs as policy
from repro.lifecycle import write_json,mark_complete,digest


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    root=tmp_path/'repo';out=root/'output';folder=out/'classic/sml_strx'
    (root/'data/processed').mkdir(parents=True);(root/'data/processed/sml.npz').write_bytes(b'data');write_json(root/'data/processed/sml.json',{})
    write_json(folder/'config.json',{});write_json(folder/'result.json',{'status':'completed'});write_json(folder/'fit_details.json',{'parameters':[.1]});mark_complete(folder)
    monkeypatch.setattr(policy,'ROOT',root);monkeypatch.setattr(policy,'OUT',out)
    monkeypatch.setattr(policy.v1,'sources',lambda:{'repairs.py':'oldhash'});monkeypatch.setattr(policy.v1,'runtime',lambda:{'statsmodels':'test'})
    cert={'training_only':True,'rule':policy.AUDIT_RULE,'source_sha256':'program','dependencies':{'repairs.py':'oldhash'},'runtime':{'statsmodels':'test'},
          'data_sha256':{'sml':digest(root/'data/processed/sml.npz')},'data_metadata_sha256':{'sml':digest(root/'data/processed/sml.json')},
          'groups':{'sml_strx':{'status':'needs_training_diagnosis','folder':'classic/sml_strx','score':.1,'boundary_violation':0.,
             'directions':[{'score':.1,'central':.1}]*3,'result_sha256':digest(folder/'result.json'),'manifest_sha256':digest(folder/'artifacts.json'),
             'parameters_sha256':digest(folder/'fit_details.json'),'data_sha256':digest(root/'data/processed/sml.npz')}}}
    def save(value):
        certpath=out/'certificate.json';write_json(certpath,value)
        policyfile=root/'configs/statistical_audit_policy.json'
        write_json(policyfile,{'version':2,'certificates':[{'path':'output/certificate.json','sha256':digest(certpath),'program_sha256':'program','rejected_groups':['sml_strx']}]})
        monkeypatch.setattr(policy,'POLICY_ROOT_SHA',digest(policyfile))
        return certpath
    return cert,folder,save


def test_only_explicit_training_rejection_can_authorize_completed_group(evidence):
    cert,_,save=evidence;policy.validate_certificate(save(cert),'sml','strx','program')
    changed=copy.deepcopy(cert);changed['groups']['sml_strx']['status']='PASS'
    with pytest.raises(ValueError):policy.validate_certificate(save(changed),'sml','strx','program')
    changed=copy.deepcopy(cert);changed['groups']['sml_strx']['score']=0.;changed['groups']['sml_strx']['directions']=[{'score':0.,'central':0.}]*3
    with pytest.raises(ValueError):policy.validate_certificate(save(changed),'sml','strx','program')


@pytest.mark.parametrize('field',['dependencies','runtime','data_sha256','data_metadata_sha256','source_sha256','training_only'])
def test_changed_audit_identity_cannot_authorize_replacement(evidence,field):
    cert,_,save=evidence;changed=copy.deepcopy(cert);changed[field]='changed'
    with pytest.raises(ValueError):policy.validate_certificate(save(changed),'sml','strx','program')


def test_audit_cannot_authorize_different_group_or_changed_original(evidence):
    cert,folder,save=evidence;certpath=save(cert)
    with pytest.raises(ValueError):policy.validate_certificate(certpath,'nasdaq','strx','program')
    (folder/'fit_details.json').write_text('{"parameters":[9.]}')
    with pytest.raises(ValueError):policy.validate_certificate(certpath,'sml','strx','program')


def test_thin_policy_preserves_v1_approval_route(tmp_path,monkeypatch):
    folder=tmp_path/'repairs/old';write_json(folder/'config.json',{})
    seen=[];monkeypatch.setattr(policy.v1,'resolve_classical',lambda *args:seen.append(args) or folder)
    assert policy.resolve_classical('sml','arimax')==folder and seen


def test_caller_cannot_supply_unpinned_self_consistent_certificate(evidence):
    cert,_,save=evidence;path=save(cert)
    changed=copy.deepcopy(cert);changed['groups']['sml_strx']['score']=.5;write_json(path,changed)
    with pytest.raises(ValueError,match='untrusted'):policy.validate_certificate(path,'sml','strx','program')


def test_rejected_canonical_cannot_count_as_strict_completion(evidence,monkeypatch):
    cert,folder,save=evidence;save(cert)
    monkeypatch.setattr(policy.v1,'resolve_classical',lambda *args:folder)
    assert policy.is_strict_complete('sml','strx') is False
    with pytest.raises(RuntimeError):policy.require_strict_complete('sml','strx')


def test_unknown_approval_version_is_not_silently_treated_as_v1(tmp_path):
    write_json(tmp_path/'config.json',{'approval_policy_version':3})
    with pytest.raises(ValueError):policy.validate_policy(tmp_path)

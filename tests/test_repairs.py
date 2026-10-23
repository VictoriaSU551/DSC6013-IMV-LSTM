"""Repair acceptance checks target provenance and future-label isolation."""
import json
import numpy as np
import pytest
from repro.repairs import replay, check_candidate, resolve_classical
from repro.lifecycle import write_json,mark_complete,digest


def fixture_data():
    rows=np.arange(2,42);x=np.sin(rows/7.)*.2+.5;y=np.sin(rows/9.)*.2+.4
    data={}
    for split,ix in [('train',np.arange(0,24)),('val',np.arange(24,32)),('test',np.arange(32,40))]:
        if split=='train':ix=ix[ix!=6]
        data[split+'_idx']=rows[ix];data[split+'_y']=y[ix]
        data[split+'_x']=np.stack([y[ix]-.01,x[ix]],axis=-1)[:,None,:]
    return data,{'features':['target','external'],'target_column':0,'depth':1}


@pytest.mark.parametrize('kind,params',[('arimax',[.01,.04,.6,.1,.01]),('strx',[.01,.001,.0001,.04])])
def test_repair_fixed_parameters_obey_one_step_causality(kind,params):
    data,info=fixture_data();pred,_=replay(data,info,kind,params,'exact_diffuse' if kind=='arimax' else 'approximate_diffuse')
    for split in ['val','test']:
        for k in [0,3,7]:
            changed=dict(data);changed[split+'_y']=data[split+'_y'].copy();changed[split+'_y'][k:]+=1000
            again,_=replay(changed,info,kind,params,'exact_diffuse' if kind=='arimax' else 'approximate_diffuse')
            np.testing.assert_array_equal(again[split][:k+1],pred[split][:k+1])


def test_replay_does_not_fit_or_estimate_test_variance(monkeypatch):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    def fail(*a,**k):raise AssertionError('test replay attempted fitting')
    monkeypatch.setattr(SARIMAX,'fit',fail)
    data,info=fixture_data();params=[.01,.04,.6,.1,.01]
    _,detail=replay(data,info,'arimax',params,'exact_diffuse')
    assert detail['parameters']==params and detail['concentrate_scale'] is False


@pytest.mark.parametrize('candidate',[
    {'status':'still_not_converged'},
    {'status':'converged_training_only_requires_replay','parameters':[float('nan')],'attempts':[{'optimizer':{'converged':True}}]},
    {'status':'converged_training_only_requires_replay','parameters':[.1],'attempts':[{'optimizer':{'converged':False}}]},
])
def test_failed_candidate_cannot_be_accepted(candidate):
    with pytest.raises(ValueError):check_candidate(candidate)


def test_resolver_rejects_path_escape(tmp_path):
    write_json(tmp_path/'approved_repairs.json',{'sml_arimax':{'folder':'../escape'}})
    with pytest.raises(ValueError,match='path'):resolve_classical('sml','arimax',tmp_path)


def test_resolver_rejects_nonstatistical_override(tmp_path):
    write_json(tmp_path/'approved_repairs.json',{'sml_rf':{'folder':'repairs/sml_rf'}})
    with pytest.raises(ValueError):resolve_classical('sml','rf',tmp_path)


def test_resolver_rejects_approval_without_commit_marker(tmp_path):
    write_json(tmp_path/'approved_repairs.json',{'sml_arimax':{'folder':'repairs/sml_arimax'}})
    (tmp_path/'repairs/sml_arimax').mkdir(parents=True)
    with pytest.raises(ValueError,match='complete'):resolve_classical('sml','arimax',tmp_path)


@pytest.mark.parametrize('gopt',[[.48],[float('nan')],[],[float('inf')]])
def test_optimizer_function_stopping_without_finite_score_is_rejected(gopt):
    candidate={'status':'converged_training_only_requires_replay','parameters':[.1],
               'initial_train_loglike':1.,'final_train_loglike':2.,
               'attempts':[{'optimizer':{'converged':True,'gopt':gopt}}]}
    with pytest.raises(ValueError,match='score'):check_candidate(candidate)


def test_successful_original_cannot_be_replaced(tmp_path):
    from repro.repairs import require_failed_original
    write_json(tmp_path/'result.json',{'status':'completed'});mark_complete(tmp_path)
    with pytest.raises(ValueError,match='successful'):require_failed_original(tmp_path)


def test_approval_is_idempotent_but_cannot_replace_another_approval():
    from repro.repairs import add_approval
    entry={'folder':'repairs/run_a','manifest_sha256':'a'};registry={'sml_arimax':entry.copy()}
    assert add_approval(registry,'sml_arimax',entry)==registry
    with pytest.raises(ValueError,match='replace'):add_approval(registry,'sml_arimax',{'folder':'repairs/run_b','manifest_sha256':'b'})


def test_candidate_generation_cannot_claim_different_data_or_environment():
    from repro.repairs import check_generation
    expected={'data_sha256':'a','data_metadata_sha256':'b','runtime':{'statsmodels':'0.14.4'},'original_evidence':{'failure.json':'c'},'dependency_sha256':{'repairs.py':'d'}}
    check_generation(expected,expected)
    for key in ['data_sha256','data_metadata_sha256','runtime','original_evidence','dependency_sha256']:
        changed=dict(expected);changed[key]='different'
        with pytest.raises(ValueError,match='generation'):check_generation(changed,expected)


@pytest.mark.parametrize('boundary',[None,float('nan'),float('inf'),1.,-1.])
def test_log_variance_candidate_requires_boundary_score_gate(boundary):
    candidate={'status':'converged_training_only_requires_replay','parameters':[.1],
               'initial_train_loglike':1.,'final_train_loglike':2.,'attempts':[{'success':True,'max_coordinate_score_per_observation':0.,'max_boundary_variance_violation':boundary}]}
    with pytest.raises(ValueError,match='boundary'):check_candidate(candidate)


def test_candidate_with_valid_score_but_wrong_final_direction_is_rejected():
    candidate={'status':'converged_training_only_requires_replay','parameters':[.1],'initial_train_loglike':1.,'final_train_loglike':2.,
               'attempts':[{'optimizer':{'converged':True,'gopt':[1e-8]}}], 'final_directional_checks':[{'score':1e-8,'central':.01}]*3}
    with pytest.raises(ValueError,match='direction'):check_candidate(candidate)
    candidate['final_directional_checks']=[{'score':1e-8,'central':1.1e-8}]*3
    np.testing.assert_equal(check_candidate(candidate),[.1])

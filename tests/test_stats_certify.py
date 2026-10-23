import numpy as np
import pytest
from repro.stats_certify import STEPS,select_plateau


def test_fixed_multistep_grid_resolves_central_difference_truncation():
    score=1e-8;curve=[{'step':h,'central':score+1e5*h*h} for h in STEPS]
    answer=select_plateau(curve,score)
    assert len(answer['plateau'])==3 and answer['plateau'][0]['step']<1e-5
    assert abs(answer['central']-score)<=2e-6


def test_one_lucky_step_or_nonfinite_curve_is_not_a_plateau():
    curve=[{'step':h,'central':.01} for h in STEPS];curve[5]['central']=0.
    with pytest.raises(ValueError):select_plateau(curve,0.)
    curve=[{'step':h,'central':float('nan')} for h in STEPS]
    with pytest.raises(ValueError):select_plateau(curve,0.)


def test_changing_the_predeclared_step_grid_is_rejected():
    curve=[{'step':h,'central':0.} for h in STEPS][1:]
    with pytest.raises(ValueError):select_plateau(curve,0.)


def test_certification_keeps_parent_warm_input_and_checks_parent_start(tmp_path):
    from repro.stats_certify import collect_parent_inputs
    from repro.lifecycle import write_json,digest
    warm=tmp_path/'warm.json';warm.write_text('{"parameters":[.1]}');parent=tmp_path/'parent'
    generation={'input_sha256':{'warm.json':digest(warm)}};setup={'source_sha256':'source','generation_provenance':generation}
    write_json(parent/'generation_start.json',{'source_sha256':'source','generation_provenance':generation})
    assert collect_parent_inputs(parent,setup,tmp_path)==[warm]
    warm.write_text('changed')
    with pytest.raises(ValueError):collect_parent_inputs(parent,setup,tmp_path)


def test_parent_start_mismatch_and_input_path_escape_are_rejected(tmp_path):
    from repro.stats_certify import collect_parent_inputs
    from repro.lifecycle import write_json
    parent=tmp_path/'parent';setup={'source_sha256':'source','generation_provenance':{'input_sha256':{}}}
    write_json(parent/'generation_start.json',{'source_sha256':'other','generation_provenance':setup['generation_provenance']})
    with pytest.raises(ValueError):collect_parent_inputs(parent,setup,tmp_path)
    setup['generation_provenance']['input_sha256']={'../outside':'any'}
    write_json(parent/'generation_start.json',setup)
    with pytest.raises(ValueError):collect_parent_inputs(parent,setup,tmp_path)

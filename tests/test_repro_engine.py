"""Checkpoint continuation must reproduce uninterrupted training, not restart it."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))


def test_resume_replays_optimizer_rng_and_epoch(tmp_path):
    import repro.engine as e
    assert hasattr(e,'train_job'), 'resumable training implementation is missing'
    rng=np.random.default_rng(0)
    arrays={}
    for s,n in [('train',32),('val',12),('test',12)]:
        x=rng.normal(size=(n,4,3)).astype('float32'); y=x[:,-1,2]*.5
        arrays.update({s+'_x':x,s+'_y':y,s+'_raw_y':y,s+'_idx':np.arange(n)})
    arrays.update(y_min=np.array(0.),y_scale=np.array(1.),x_min=np.zeros(3),x_scale=np.ones(3))
    data=tmp_path/'tiny.npz'; np.savez(data,**arrays)
    info={'features':['a','b','target'],'target_column':2,'depth':4,'protocol':'test'}
    import json
    data.with_suffix('.json').write_text(json.dumps(info))
    cfg=dict(dataset='tiny',model='tensor',objective='em',units=2,lr=.001,dropout=.2,l2=.0001,
             seed=11,epochs=3,patience=99,batch_size=8,device='cpu',threads=1)
    continuous=e.train_job(cfg,tmp_path/'continuous',data,final=False)
    partial=e.train_job(cfg,tmp_path/'resumed',data,final=False,stop_after=1)
    assert partial['status']=='interrupted'
    resumed=e.train_job(cfg,tmp_path/'resumed',data,final=False)
    assert continuous['val_rmse_scaled']==resumed['val_rmse_scaled']
    a=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert b['epoch']==3
    for k,v in a['model'].items():
        torch.testing.assert_close(v,b['model'][k],rtol=0,atol=0)


def test_result_without_commit_marker_is_reconstructed_and_config_change_rejected(tmp_path):
    import repro.engine as e
    import json
    rng=np.random.default_rng(5); d={}
    for split,n in [('train',16),('val',8),('test',8)]:
        x=rng.normal(size=(n,3,2)).astype('float32');y=x[:,-1,1]
        d.update({split+'_x':x,split+'_y':y,split+'_raw_y':y,split+'_idx':np.arange(n)})
    d.update(y_min=np.array(0.),y_scale=np.array(1.),x_min=np.zeros(2),x_scale=np.ones(2))
    path=tmp_path/'tiny.npz';np.savez(path,**d)
    path.with_suffix('.json').write_text(json.dumps({'features':['a','target'],'target_column':1,'depth':3,'protocol':'test'}))
    cfg=dict(dataset='tiny',model='tensor',objective='em',units=2,lr=.001,dropout=0.,l2=.0001,seed=11,
             epochs=1,patience=9,batch_size=8,device='cpu',threads=1)
    folder=tmp_path/'run'
    result=e.train_job(cfg,folder,path,final=False)
    (folder/'artifacts.json').unlink();(folder/'complete.json').unlink()
    restored=e.train_job(cfg,folder,path,final=False)
    assert restored['val_rmse_scaled']==result['val_rmse_scaled']
    assert (folder/'complete.json').exists()
    import pytest
    with pytest.raises(ValueError): e.train_job({**cfg,'units':3},folder,path,final=False)

"""Resumable, validation-selected one-step forecasting with durable evidence."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from .models import build_model, mixture_statistics
from .lifecycle import is_complete, mark_complete, validate_config, manifest

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def atomic_torch(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp); tmp.replace(path)


def metrics(y,pred):
    e=np.asarray(pred,dtype='float64')-np.asarray(y,dtype='float64')
    return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))}


def load_data(path,selected=None):
    path=Path(path)
    info=json.loads(path.with_suffix('.json').read_text())
    if 'npz_sha256' in info and sha(path)!=info['npz_sha256']:
        raise ValueError('data checksum mismatch')
    with np.load(path) as z: arrays={k:z[k] for k in z.files}
    all_features=info['features']
    selected=list(range(len(all_features))) if selected is None else list(selected)
    if len(set(selected))!=len(selected) or not selected or min(selected)<0 or max(selected)>=len(all_features):
        raise ValueError('invalid selected feature indices')
    original_target=info['target_column']
    for split in ['train','val','test']:
        arrays[split+'_target_history']=arrays[split+'_x'][:,:,original_target].copy()
        arrays[split+'_x']=arrays[split+'_x'][:,:,selected]
    info={**info,'original_features':all_features,'features':[all_features[i] for i in selected],
          'selected_indices':selected,'target_column':selected.index(original_target) if original_target in selected else None}
    return arrays,info


def model_forward(model,x,target_history):
    if getattr(model,'target','not_dual') is None or model.__class__.__name__=='DUAL':
        return model(x,target_history=target_history)
    return model(x)


@torch.no_grad()
def evaluate(model,x,y,target_history,objective='mse',batch=512,explain=False):
    model.eval(); preds=[]; q_sum=prior_sum=alpha_sum=None; n=0
    for start in range(0,len(y),batch):
        stop=start+batch
        out=model_forward(model,x[start:stop],target_history[start:stop])
        p=out['prediction']
        if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite prediction')
        preds.append(p.cpu().numpy())
        if explain:
            if 'mu' in out and objective=='em':
                q=mixture_statistics(out['mu'],out['sigma'],out['logits'],y[start:stop])['posterior']
            else: q=out['importance']
            prior=out['importance']
            qq=q.double().sum(0).cpu().numpy(); pp=prior.double().sum(0).cpu().numpy()
            q_sum=qq if q_sum is None else q_sum+qq
            prior_sum=pp if prior_sum is None else prior_sum+pp
            if 'alpha' in out:
                a=out['alpha'].double().sum(0).cpu().numpy()
                alpha_sum=a if alpha_sum is None else alpha_sum+a
            n+=len(p)
    result={'prediction':np.concatenate(preds)}
    if explain:
        result.update(importance=q_sum/n,prior=prior_sum/n,temporal=None if alpha_sum is None else alpha_sum/n)
    return result


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def file_manifest(folder):
    return manifest(folder)


def fingerprint_config(config,data_path,final):
    return {**config,'final':final,'data_sha256':sha(data_path),
            'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('lifecycle.py')]}}


def train_job(config,folder,data_path,final=True,stop_after=None):
    """Resume exactly from last.pt; search runs never evaluate the test split."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    config=dict(config); data_path=Path(data_path)
    atomic_config=fingerprint_config(config,data_path,final)
    validate_config(folder/'config.json',atomic_config)
    if is_complete(folder):
        return json.loads((folder/'result.json').read_text())
    threads=int(config.get('threads',2)); torch.set_num_threads(threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(.6)
    d,info=load_data(data_path,config.get('selected'))
    n=len(info['features'])
    model=build_model(config['model'],n,config['units'],info['depth'],info['target_column'],
                      config.get('dropout',0.),config.get('sigma_floor',.001)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    # The held-out split is left on CPU and is not evaluated during search or fitting.
    tensors={s:{k:torch.from_numpy(np.asarray(d[s+'_'+k],dtype='float32')).to(device)
                for k in ['x','y','target_history']} for s in ['train','val']}
    start=time.monotonic(); elapsed=0.; best=float('inf'); best_epoch=0; bad=0; begin=1; history=[]; i_history=[]
    if (folder/'last.pt').exists():
        ck=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        begin=ck['epoch']+1; best=ck['best']; best_epoch=ck['best_epoch']; bad=ck['bad']; elapsed=ck['seconds']
        history=ck['history']; i_history=ck['importance_history']; restore_rng(ck['rng'])
    train=tensors['train']; val=tensors['val']; objective=config.get('objective','mse')
    count=len(train['y']); batch=config.get('batch_size',64)
    for epoch in range(begin,config['epochs']+1):
        if bad>=config['patience']: break
        epoch_start=time.monotonic(); model.train(); sq=loss_sum=0.; online_q=np.zeros(n); online_prior=np.zeros(n)
        order=torch.randperm(count)
        for offset in range(0,count,batch):
            ix=order[offset:offset+batch].to(device)
            x=train['x'][ix]; y=train['y'][ix]
            optimizer.zero_grad(set_to_none=True)
            out=model_forward(model,x,train['target_history'][ix]); prediction=out['prediction']
            if objective=='em':
                statistics=mixture_statistics(out['mu'],out['sigma'],out['logits'],y)
                loss=statistics['em_loss']; q=statistics['posterior']
            else:
                loss=(prediction-y).square().mean(); q=out['importance'].detach()
            if config.get('l2',0):
                penalty=sum(p.square().sum() for name,p in model.named_parameters() if p.ndim>=2)
                loss=loss+.5*config['l2']*penalty
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            size=len(ix); sq+=float((prediction.detach()-y).square().sum()); loss_sum+=float(loss.detach())*size
            online_q+=q.double().sum(0).detach().cpu().numpy()
            online_prior+=out['importance'].double().sum(0).detach().cpu().numpy()
        vp=evaluate(model,**val,objective=objective)['prediction']
        score=metrics(d['val_y'],vp)['rmse']
        if not math.isfinite(score): raise FloatingPointError('nonfinite validation score')
        seconds=elapsed+time.monotonic()-start
        history.append({'epoch':epoch,'train_rmse_scaled':math.sqrt(sq/count),'train_loss':loss_sum/count,
                        'val_rmse_scaled':score,'epoch_seconds':time.monotonic()-epoch_start,'seconds':seconds})
        i_history.append({'epoch':epoch,'importance':(online_q/count).tolist(),'prior':(online_prior/count).tolist()})
        if score<best:
            best=score; best_epoch=epoch; bad=0
            atomic_torch(folder/'best.pt',{'model':model.state_dict(),'epoch':epoch,'config':atomic_config})
        else: bad+=1
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,
                    'best_epoch':best_epoch,'bad':bad,'seconds':seconds,'history':history,'importance_history':i_history,'rng':rng_state()}
        atomic_torch(folder/'last.pt',checkpoint)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        atomic_json(folder/'importance_history.json',i_history)
        atomic_json(folder/'status.json',{'status':'training','epoch':epoch,'epochs':config['epochs'],
                                         'best_epoch':best_epoch,'best_val_rmse_scaled':best,'seconds':seconds})
        print(json.dumps({'run':folder.name,'epoch':epoch,'val_rmse':score,'best':best}),flush=True)
        if stop_after is not None and epoch>=stop_after: return {'status':'interrupted','epoch':epoch}
    ck=torch.load(folder/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    result={'status':'completed','dataset':config['dataset'],'model':config['model'],'objective':objective,
            'seed':config['seed'],'selection':config.get('selection','all'),'selected':info['selected_indices'],
            'best_epoch':best_epoch,'epochs_completed':len(history),'val_rmse_scaled':best,
            'parameters':sum(p.numel() for p in model.parameters()),
            'recurrent_parameters':model.recurrent_parameter_count() if hasattr(model,'recurrent_parameter_count') else None,
            'seconds':elapsed+time.monotonic()-start,'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'numpy':np.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}}
    if final:
        explanation=evaluate(model,**train,objective=objective,explain=True)
        rank=pd.DataFrame({'feature':info['features'],'original_index':info['selected_indices'],
                           'importance':explanation['importance'],'prior_attention':explanation['prior']})
        rank.to_csv(folder/'importance.csv',index=False)
        if explanation['temporal'] is not None:
            np.savez_compressed(folder/'temporal.npz',temporal=explanation['temporal'])
        test={k:torch.from_numpy(np.asarray(d['test_'+k],dtype='float32')).to(device) for k in ['x','y','target_history']}
        pred=evaluate(model,**test,objective=objective)['prediction']
        raw_pred=pred.astype('float64')*float(d['y_scale'])+float(d['y_min'])
        raw_y=d['test_raw_y']; persistence=d['test_target_history'][:,-1].astype('float64')
        # Target x and y scalers may differ at the last training boundary.
        original_target=info['original_features'].index(json.loads(data_path.with_suffix('.json').read_text())['target']) if 'target' in json.loads(data_path.with_suffix('.json').read_text()) else json.loads(data_path.with_suffix('.json').read_text())['target_column']
        persistence=persistence*float(d['x_scale'][original_target])+float(d['x_min'][original_target])
        result.update(test_raw=metrics(raw_y,raw_pred),test_scaled=metrics(d['test_y'],pred),
                      persistence_raw=metrics(raw_y,persistence),test_n=len(raw_y))
        pd.DataFrame({'target_row':d['test_idx'],'y_true':raw_y,'y_pred':raw_pred,'persistence':persistence,
                      'y_true_scaled':d['test_y'],'y_pred_scaled':pred}).to_csv(folder/'predictions.csv',index=False)
        # Global importance is the training-instance posterior average at selected checkpoint.
        result['importance_definition']='Eq11 posterior on training labels' if objective=='em' else 'attention/contribution (not EM posterior)'
    atomic_json(folder/'result.json',result)
    atomic_json(folder/'status.json',{'status':'completed','epoch':len(history),'best_epoch':best_epoch})
    mark_complete(folder)
    del model,optimizer,tensors
    if device.type=='cuda': torch.cuda.empty_cache()
    return result

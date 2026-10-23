import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from repro.verify import verify_classical_alignment


def test_shifted_classical_target_rows_are_rejected_even_if_metrics_match():
    data={'test_idx':np.array([10,11,12]),'test_raw_y':np.array([1.,2.,3.]),'test_y':np.array([.1,.2,.3])}
    frame=pd.DataFrame({'target_row':[10,11,12],'y_true':[1.,2.,3.],'y_true_scaled':[.1,.2,.3]})
    verify_classical_alignment(frame,data)
    frame['target_row']+=1
    with pytest.raises(AssertionError): verify_classical_alignment(frame,data)


def test_classical_wrong_labels_or_partial_exports_are_rejected():
    data={'test_idx':np.array([10,11,12]),'test_raw_y':np.array([1.,2.,3.]),'test_y':np.array([.1,.2,.3])}
    frame=pd.DataFrame({'target_row':[10,11,12],'y_true':[2.,3.,4.],'y_true_scaled':[.1,.2,.3]})
    with pytest.raises(AssertionError): verify_classical_alignment(frame,data)
    frame['y_true']=[1.,2.,3.]
    with pytest.raises(AssertionError): verify_classical_alignment(frame.iloc[:2],data)


def test_state_space_current_or_future_test_targets_do_not_change_earlier_predictions():
    from repro.classical import state_space
    rng=np.random.default_rng(77); data={}; start=2
    for split,n in [('train',45),('val',12),('test',12)]:
        x=rng.normal(size=(n,2,3)).astype('float32');y=rng.normal(size=n).astype('float32')
        data[split+'_x']=x;data[split+'_y']=y;data[split+'_idx']=np.arange(start,start+n);start+=n
    info={'features':['a','b','target'],'target_column':2,'depth':2}
    first,_=state_space(data,info,'strx')
    data['test_y']=data['test_y'].copy();data['test_y'][-1]+=100
    second,_=state_space(data,info,'strx')
    # Last observed y itself is not part of its one-step prediction either.
    np.testing.assert_allclose(first['test'],second['test'],rtol=0,atol=1e-10)


@pytest.mark.parametrize('kind,dtype',[('xgt',np.float32),('enet',np.float32),('rf',np.float64)])
def test_csv_precision_reconstruction_retains_original_metric_gate(kind,dtype):
    import io
    from repro.engine import metrics
    from repro.verify import verify_classical_metrics
    data={'test_y':np.array([.123456789,.345678912,.567891234],dtype=np.float32),
          'test_raw_y':np.array([3345.67891,3400.12345,3460.98765]),'y_scale':262.347,'y_min':3312.718}
    pred=np.array([.123556789,.345778912,.567991234],dtype=dtype);raw=pred*data['y_scale']+data['y_min']
    frame=pd.read_csv(io.StringIO(pd.DataFrame({'y_pred_scaled':pred,'y_pred':raw,'y_true_scaled':data['test_y'],'y_true':data['test_raw_y']}).to_csv(index=False)))
    row={'test_scaled':metrics(data['test_y'],pred),'test_raw':metrics(data['test_raw_y'],raw)}
    verify_classical_metrics(frame,data,row,kind)
    bad={key:dict(value) for key,value in row.items()};bad['test_raw']['rmse']+=.001
    with pytest.raises(AssertionError):verify_classical_metrics(frame,data,bad,kind)
    frame.loc[0,'y_pred']+=1.
    with pytest.raises(AssertionError):verify_classical_metrics(frame,data,row,kind)

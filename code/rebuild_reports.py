"""Rebuild seven numerical CSVs and two figures from retained final artifacts.

Writes to a new directory only. The reviewed narrative report is not replaced.
"""
import argparse
import json
import math
from pathlib import Path
from windows_compat import install
install()
import pandas as pd
import final_package as p
from repro import report
from repro.engine import load_data

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/reproduction'

def collect():
    rows=[]
    for name in p.expected_deep():
        folder=OUT/name;r=p.read(folder/'result.json');cfg=p.read(folder/'config.json')
        scale=float(load_data(ROOT/f"data/processed/{r['dataset']}.npz")[0]['y_scale'])
        rows.append({**{key:r[key] for key in ['dataset','model','objective','selection','seed']},
                     **{metric+'_'+unit:r['test_'+unit][metric] for unit in ['raw','scaled'] for metric in ['rmse','mae']},
                     'persistence_rmse_raw':r['persistence_raw']['rmse'],'persistence_rmse_scaled':r['persistence_raw']['rmse']/scale,
                     'best_epoch':r['best_epoch'],'epochs':r['epochs_completed'],'parameters':r['parameters'],'seconds':r['seconds'],
                     **{key:cfg[key] for key in ['units','lr','dropout','l2']},'features':len(r['selected']),'status':r['status'],'run':folder.name})
    for ds in p.DATASETS:
        for kind in ['rf','xgt','enet','strx','arimax']:
            r=p.read(OUT/'classic'/f'{ds}_{kind}'/'result.json')
            for row in r['runs']:
                rows.append({'dataset':ds,'model':kind,'objective':'classical','selection':'all','seed':row['seed'],
                             **{metric+'_'+unit:row['test_'+unit][metric] for unit in ['raw','scaled'] for metric in ['rmse','mae']},
                             'seconds':r['seconds'],'status':r['status'],'run':f'{ds}_{kind}',
                             'statistical_repair':r.get('repair',False),'initialization':r.get('initialization','original default')})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',required=True)
    args=parser.parse_args();destination=Path(args.destination).resolve()
    if destination.exists():raise ValueError('destination exists; choose a new folder')
    if destination==ROOT or (ROOT/'reports').resolve() in destination.parents:raise ValueError('cannot overwrite final project/report')
    manifest=p.read(ROOT/'reports/evidence/final_manifest.json');p.verify_files(ROOT,manifest['files'])
    destination.mkdir(parents=True);(destination/'figures').mkdir()
    runs=collect();agg=report.summary(runs)
    runs.to_csv(destination/'seed_metrics.csv',index=False);agg.to_csv(destination/'summary_metrics.csv',index=False)
    agg[agg.selection!='all'].to_csv(destination/'variable_selection.csv',index=False)
    uncertain,stable=report.uncertainty_and_stability()
    uncertain.to_csv(destination/'uncertainty.csv',index=False);stable.to_csv(destination/'interpretation_stability.csv',index=False)
    data=[]
    for ds in p.DATASETS:
        info=p.read(ROOT/f'data/processed/{ds}.json')
        data.append({'dataset':ds,'raw_rows':info['raw_rows'],'variables':len(info['features']),'window':info['depth'],
                     **{s:info['splits'][s]['n'] for s in ['train','val','test']},'dropped_windows':info['invalid_or_segment_crossing_windows']})
    pd.DataFrame(data).to_csv(destination/'data_audit.csv',index=False)
    paper=[]
    for (ds,model),(rmse,mae) in report.PAPER.items():
        ours=agg[(agg.dataset==ds)&(agg.model==model)&(agg.objective=='em')&(agg.selection=='all')].iloc[0]
        paper.append({'dataset':ds,'model':model,'paper_rmse':rmse,'paper_mae':mae,'our_rmse_raw':ours.rmse_raw_mean,
                      'our_rmse_scaled':ours.rmse_scaled_mean,'directly_comparable':False,
                      'reason':'paper scaling and/or exact preprocessing not fully specified'})
    pd.DataFrame(paper).to_csv(destination/'paper_comparison.csv',index=False)
    report.DELIVERY=destination;report.plots(agg,runs)
    # Compare all numerical outputs with reviewed published tables, allowing only decimal serialization noise.
    from pandas.testing import assert_frame_equal
    for path in destination.glob('*.csv'):
        rebuilt=pd.read_csv(path);saved=pd.read_csv(ROOT/'reports'/path.name)
        assert_frame_equal(rebuilt,saved,check_exact=False,rtol=1e-11,atol=1e-12)
    print('REPORT_REBUILD_PASS: 7 CSVs reconciled and 2 figures rebuilt at',destination)

if __name__=='__main__':main()

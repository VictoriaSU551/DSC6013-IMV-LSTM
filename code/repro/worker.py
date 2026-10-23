"""One isolated worker invocation per registered unit."""
import argparse
import json
from pathlib import Path
import traceback
from .engine import ROOT,atomic_json,train_job
from .lifecycle import run_lock


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('job'); args=parser.parse_args()
    job=json.loads(Path(args.job).read_text()); folder=Path(job['folder'])
    lock=run_lock(folder)  # Held by worker, so an orphan is protected after supervisor death.
    try:
        if job['type']=='deep':
            train_job(job['config'],folder,ROOT/'data/processed'/f"{job['config']['dataset']}.npz",final=job['final'])
        else:
            from .classical import run_classical
            run_classical(job['dataset'],job['model'],job['seeds'],folder)
    except Exception as exc:
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__': main()

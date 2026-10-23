"""Stdlib-only transactional completion and worker-level exclusion."""
import fcntl
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)


def validate_config(path,expected):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise ValueError('run configuration/source/data changed; preserve evidence and use new run ID')
    else: write_json(path,expected)


def manifest(folder):
    excluded={'artifacts.json','complete.json','failure.json','worker.lock'}
    return {p.name:digest(p) for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.name not in excluded and not p.name.endswith('.tmp')}


def mark_complete(folder):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text())
    write_json(folder/'artifacts.json',manifest(folder))
    # Commit marker is last. Missing marker means resume/reconstruct, never reuse.
    write_json(folder/'complete.json',{'status':result['status'],'result_sha256':digest(folder/'result.json'),
                                     'manifest_sha256':digest(folder/'artifacts.json')})


def is_complete(folder,require_converged=True):
    folder=Path(folder)
    if not (folder/'complete.json').exists(): return False
    marker=json.loads((folder/'complete.json').read_text())
    for name,key in [('result.json','result_sha256'),('artifacts.json','manifest_sha256')]:
        if not (folder/name).exists() or digest(folder/name)!=marker[key]:
            raise ValueError('completion integrity mismatch: '+str(folder/name))
    for name,want in json.loads((folder/'artifacts.json').read_text()).items():
        if not (folder/name).exists() or digest(folder/name)!=want:
            raise ValueError('artifact integrity mismatch: '+str(folder/name))
    return not require_converged or marker['status']=='completed'


def run_lock(folder,blocking=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/'worker.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX if blocking else fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle

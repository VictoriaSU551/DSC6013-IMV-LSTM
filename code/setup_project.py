"""Fetch hash-pinned source/data; apply a minimal device compatibility patch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from common import ROOT, save_json, sha256


def fetch_asset(asset):
    path = ROOT / asset['path']
    if path.exists():
        if sha256(path) != asset['sha256']:
            raise ValueError(f'Existing file differs from lock: {asset["path"]}')
        return
    for attempt in range(3):
        try:
            with urllib.request.urlopen(asset['url'], timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != asset['sha256']:
                raise ValueError(f'Download hash mismatch: {asset["path"]}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def patch_model():
    source = ROOT / 'upstream/IMV_LSTM_pytorch/networks.py'
    text = source.read_text(encoding='utf-8')
    # No formula, initialisation, parameter name or shape changes.
    text = 'from typing import List\nfrom torch import Tensor\n' + text
    assert text.count('torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()') == 3
    assert text.count('torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()') == 1
    text = text.replace('torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()',
                        'x.new_zeros((x.shape[0], self.input_dim, self.n_units))')
    text = text.replace('torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()',
                        'x.new_zeros((x.shape[0], self.input_dim*self.n_units))')
    # Modern einsum may return non-contiguous storage; preserve logical flatten order.
    assert text.count('.view(') == 3
    text = text.replace('.view(', '.reshape(')
    path = ROOT / 'code/vendor/networks.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true', help='Skip data downloads (CI).')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'sources.lock.json').read_text(encoding='utf-8'))
    assets = [a for a in lock['assets'] if not args.source_only or a['kind'] == 'source']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_asset, assets))
    patched = patch_model()
    save_json(ROOT / 'output/setup_verification.json',
              {'status': 'PASS', 'files_verified': len(assets), 'patched_model_sha256': patched,
               'lock_sha256': sha256(ROOT / 'sources.lock.json')})
    print(f'Verified {len(assets)} assets; model CPU patch ready.')


if __name__ == '__main__':
    main()

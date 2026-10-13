"""Independently reload checkpoints and recompute saved predictions/metrics."""
import argparse
import json
import numpy as np
import pandas as pd
import torch

from common import ROOT, save_json, sha256
from model import build_model
from prepare_data import read_raw


def verify_data(dataset):
    path = ROOT/'data/processed'/f'{dataset}.npz'
    info = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha256(path) == info['npz_sha256']
    assert sha256(ROOT/'code/prepare_data.py') == info['preparation_code_sha256']
    for rel, digest in info['raw_sha256'].items():
        assert sha256(ROOT/rel) == digest
    d = dict(np.load(path))
    raw, features, target, cuts, segments, _ = read_raw(dataset)
    train_idx, val_idx, test_idx = (d[s+'_idx'] for s in ['train', 'val', 'test'])
    assert train_idx[-1] < cuts[0] <= val_idx[0] <= val_idx[-1] < cuts[1] <= test_idx[0]
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx.tolist()))
    for split in ['train', 'val', 'test']:
        idx = d[split+'_idx']
        expected_x = np.stack([raw[i-info['depth']:i] for i in idx])
        if segments is not None:
            assert all(np.all(segments[i-info['depth']:i+1] == segments[i]) for i in idx)
        assert np.isfinite(expected_x).all() and np.isfinite(raw[idx, target]).all()
        np.testing.assert_array_equal(raw[idx, target], d[split+'_raw_y'])
        np.testing.assert_allclose(d[split+'_x'], (expected_x-d['x_min'])/d['x_scale'], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(d[split+'_y'], (raw[idx, target]-d['y_min'])/d['y_scale'], atol=1e-6)
        if split == 'train':
            lo, hi = expected_x.min((0, 1)), expected_x.max((0, 1))
            np.testing.assert_array_equal(d['x_min'], lo)
            np.testing.assert_array_equal(d['x_scale'], np.where(hi > lo, hi-lo, 1.))
            assert d['y_min'] == raw[idx, target].min()
            span = raw[idx, target].max()-raw[idx, target].min()
            assert d['y_scale'] == (span if span > 0 else 1.)
    return d, info


def check_metric(actual, truth, pred):
    error = truth.astype(float)-pred.astype(float)
    np.testing.assert_allclose(actual['rmse'], np.sqrt(np.mean(error**2)), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual['mae'], np.mean(np.abs(error)), rtol=1e-7, atol=1e-9)


@torch.no_grad()
def verify_run(folder, d, info):
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    result = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    artifacts = json.loads((folder/'artifacts.json').read_text(encoding='utf-8'))
    for name, digest in artifacts.items():
        assert sha256(folder/name) == digest, f'Artifact changed: {folder/name}'
    for name, digest in config['source_sha256'].items():
        assert sha256(ROOT/name) == digest, f'Code changed: {name}'
    assert config['data_sha256'] == sha256(ROOT/'data/processed'/f'{config["dataset"]}.npz')
    assert config['metadata_sha256'] == sha256(ROOT/'data/processed'/f'{config["dataset"]}.json')
    model = build_model(config['model'], config['input_dim'], config['units'])
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    history = pd.read_csv(folder/'history.csv')
    assert int(history.loc[history['val_rmse_scaled'].idxmin(), 'epoch']) == checkpoint['epoch'] == result['best_epoch']
    assert len(history) == result['epochs_completed']
    attention = np.load(folder/'attention.npz')
    for split in ['val', 'test', 'train']:
        predictions, alphas, betas = [], [], []
        for start in range(0, len(d[split+'_x']), config['batch_size']):
            x = torch.from_numpy(d[split+'_x'][start:start+config['batch_size']])
            pred, alpha, beta = model(x)
            predictions.append(pred[:, 0].numpy())
            a, b = alpha[..., 0].numpy(), beta[..., 0].numpy()
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a.sum(1), 1., atol=1e-5)
            np.testing.assert_allclose(b.sum(1), 1., atol=1e-5)
            alphas.append(a.astype(float).sum(0))
            betas.append(b.astype(float).sum(0))
        pred = np.concatenate(predictions)
        if split == 'val':
            np.testing.assert_allclose(np.sqrt(np.mean((pred.astype(float)-d['val_y'])**2)), result['best_val_rmse_scaled'], rtol=1e-6)
            continue
        np.testing.assert_allclose(np.sum(alphas, axis=0)/len(pred), attention[split+'_temporal'], atol=1e-7)
        np.testing.assert_allclose(np.sum(betas, axis=0)/len(pred), attention[split+'_variable'], atol=1e-7)
        if split == 'train':
            table = pd.read_csv(folder/'variable_attention.csv')
            assert table.feature.tolist() == info['features']
            np.testing.assert_allclose(table.mean_train_variable_attention, attention['train_variable'], atol=1e-7)
        if split == 'test':
            table = pd.read_csv(folder/'predictions.csv')
            assert len(table) == result['test_n']
            np.testing.assert_array_equal(table.target_row, d['test_idx'])
            np.testing.assert_allclose(table.y_true, d['test_raw_y'], atol=1e-8)
            np.testing.assert_allclose(table.y_true_scaled, d['test_y'], atol=1e-8)
            np.testing.assert_allclose(table.y_pred_scaled, pred, atol=1e-7)
            raw_pred = pred.astype(float)*d['y_scale']+d['y_min']
            np.testing.assert_allclose(table.y_pred, raw_pred, rtol=1e-7, atol=1e-7)
            baseline = d['test_x'][:, -1, info['target_column']].astype(float)*d['x_scale'][info['target_column']]+d['x_min'][info['target_column']]
            np.testing.assert_allclose(table.persistence, baseline, rtol=1e-7, atol=1e-7)
            check_metric(result['test_raw'], d['test_raw_y'], raw_pred)
            check_metric(result['test_scaled'], d['test_y'], pred)
            check_metric(result['persistence_raw'], d['test_raw_y'], baseline)
    return {'run': folder.relative_to(ROOT).as_posix(), 'status': 'PASS', 'test_n': result['test_n'],
            'checkpoint_sha256': sha256(folder/'best.pt'), **{k: result[k] for k in ['dataset', 'model', 'best_epoch', 'test_raw', 'test_scaled', 'persistence_raw']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', default='pipeline_10epochs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_root = ROOT/'output/runs'/args.run_tag
    folders = sorted(run_root.glob('*/metrics.json'))
    if not folders:
        raise ValueError('No completed runs')
    expected = set()
    for config_file in run_root.glob('*/config.json'):
        config = json.loads(config_file.read_text(encoding='utf-8'))
        expected.update(f'{dataset}_{kind}_seed{config["seed"]}'
                        for dataset in config['datasets'] for kind in config['models'])
    completed = {file.parent.name for file in folders}
    if expected != completed:
        raise ValueError(f'Incomplete experiment matrix; missing={sorted(expected-completed)}, unexpected={sorted(completed-expected)}')
    cache, results = {}, []
    for file in folders:
        name = json.loads(file.read_text())['dataset']
        if name not in cache:
            cache[name] = verify_data(name)
        results.append(verify_run(file.parent, *cache[name]))
        print(file.parent.name, 'PASS', flush=True)
    save_json(ROOT/'output'/f'verification_{args.run_tag}.json',
              {'status': 'PASS', 'scope': 'PIPELINE_ONLY_NOT_FULL_PAPER', 'runs': results})
    pd.DataFrame([{'dataset': r['dataset'], 'model': r['model'], 'best_epoch': r['best_epoch'],
                   'test_n': r['test_n'], 'rmse_raw': r['test_raw']['rmse'],
                   'mae_raw': r['test_raw']['mae'], 'rmse_scaled': r['test_scaled']['rmse'],
                   'mae_scaled': r['test_scaled']['mae'],
                   'persistence_rmse_raw': r['persistence_raw']['rmse'], 'verified': r['status']}
                  for r in results]).to_csv(ROOT/'output'/f'metrics_{args.run_tag}.csv', index=False)


if __name__ == '__main__':
    main()

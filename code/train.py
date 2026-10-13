"""Train both upstream models; choose checkpoints using validation only."""
import argparse
import json
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, save_json, sha256
from model import build_model


def metrics(y, pred):
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error)))}


def loader(d, split, batch, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(d[split+'_x']), torch.from_numpy(d[split+'_y'])),
                      batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    predictions, targets = [], []
    alpha_sum = beta_sum = None
    count = 0
    for x, y in batches:
        pred, alpha, beta = model(x.to(device))
        predictions.append(pred[:, 0].cpu().numpy())
        targets.append(y.numpy())
        a, b = alpha.cpu().numpy()[..., 0], beta.cpu().numpy()[..., 0]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.all(a >= 0) and np.all(b >= 0)
        assert np.allclose(a.sum(axis=1), 1., atol=1e-5)
        assert np.allclose(b.sum(axis=1), 1., atol=1e-5)
        aa, bb = a.astype(float).sum(0), b.astype(float).sum(0)
        alpha_sum = aa if alpha_sum is None else alpha_sum+aa
        beta_sum = bb if beta_sum is None else beta_sum+bb
        count += len(y)
    return np.concatenate(predictions), np.concatenate(targets), alpha_sum/count, beta_sum/count


def run(args, dataset, kind):
    folder = ROOT/'output/runs'/args.run_tag/f'{dataset}_{kind}_seed{args.seed}'
    if folder.exists():
        raise FileExistsError(f'Run already exists: {folder}. Choose a new --run-tag; existing evidence is preserved.')
    data_path = ROOT/'data/processed'/f'{dataset}.npz'
    metadata_path = data_path.with_suffix('.json')
    info = json.loads(metadata_path.read_text(encoding='utf-8'))
    if sha256(data_path) != info['npz_sha256']:
        raise ValueError('Processed data hash mismatch')
    with np.load(data_path) as archive:
        d = dict(archive)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = build_model(kind, len(info['features']), args.units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = loader(d, 'train', args.batch_size, True)
    val_loader = loader(d, 'val', args.batch_size)
    config = {**vars(args), 'dataset': dataset, 'model': kind, 'loss': 'MSE',
              'protocol': info['protocol'], 'status_scope': 'PIPELINE_ONLY_NOT_FULL_PAPER',
              'input_dim': len(info['features']), 'data_sha256': sha256(data_path),
              'metadata_sha256': sha256(metadata_path),
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p)
                                for p in [ROOT/'code/vendor/networks.py', ROOT/'code/train.py',
                                          ROOT/'code/model.py', ROOT/'code/common.py', ROOT/'code/prepare_data.py']},
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'numpy': np.__version__, 'pandas': pd.__version__}}
    folder.mkdir(parents=True)
    save_json(folder/'config.json', config)
    best, bad, history = float('inf'), 0, []
    start = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        squared, n = 0., 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, _, _ = model(x.to(device))
            loss = (pred[:, 0]-y.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            squared += loss.item()*len(y)
            n += len(y)
        pred, y, _, _ = evaluate(model, val_loader, device)
        val_rmse = metrics(y, pred)['rmse']
        if not np.isfinite(val_rmse):
            raise ValueError('Non-finite validation metric')
        history.append({'epoch': epoch, 'train_rmse_scaled': (squared/n)**.5,
                        'val_rmse_scaled': val_rmse, 'seconds': time.monotonic()-start})
        pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
        if val_rmse < best:
            best, best_epoch, bad = val_rmse, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch}, folder/'best.pt')
        else:
            bad += 1
        print(f'{dataset}/{kind} {epoch}/{args.epochs} train={history[-1]["train_rmse_scaled"]:.6f} val={val_rmse:.6f}', flush=True)
        if bad >= args.patience:
            break
    checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    pred, truth, alpha, beta = evaluate(model, loader(d, 'test', args.batch_size), device)
    # Aggregate explanations on training instances, not test labels.
    _, _, train_alpha, train_beta = evaluate(model, loader(d, 'train', args.batch_size), device)
    raw_pred = pred.astype(float)*float(d['y_scale'])+float(d['y_min'])
    raw_truth = d['test_raw_y']
    baseline = d['test_x'][:, -1, info['target_column']].astype(float)*d['x_scale'][info['target_column']]+d['x_min'][info['target_column']]
    pd.DataFrame({'target_row': d['test_idx'], 'y_true': raw_truth, 'y_pred': raw_pred,
                  'persistence': baseline, 'y_true_scaled': truth, 'y_pred_scaled': pred}).to_csv(folder/'predictions.csv', index=False)
    np.savez_compressed(folder/'attention.npz', train_temporal=train_alpha, train_variable=train_beta,
                        test_temporal=alpha, test_variable=beta)
    pd.DataFrame({'feature': info['features'], 'mean_train_variable_attention': train_beta}).to_csv(folder/'variable_attention.csv', index=False)
    result = {'dataset': dataset, 'model': kind, 'seed': args.seed, 'best_epoch': best_epoch,
              'epochs_completed': len(history), 'test_n': len(truth), 'best_val_rmse_scaled': best,
              'test_raw': metrics(raw_truth, raw_pred), 'test_scaled': metrics(truth, pred),
              'persistence_raw': metrics(raw_truth, baseline), 'seconds': time.monotonic()-start,
              'status': 'PIPELINE_ONLY_NOT_FULL_PAPER'}
    save_json(folder/'metrics.json', result)
    save_json(folder/'artifacts.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['sml', 'pm25', 'nasdaq'], default=['sml', 'pm25', 'nasdaq'])
    parser.add_argument('--models', nargs='+', choices=['tensor', 'full'], default=['tensor', 'full'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--units', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--run-tag', default='pipeline_10epochs')
    args = parser.parse_args()
    if any(v <= 0 for v in [args.epochs, args.units, args.batch_size, args.patience, args.threads, args.lr]):
        parser.error('Training parameters must be positive')
    if not args.run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_tag):
        parser.error('--run-tag must contain only letters, digits, _ or -')
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        for kind in args.models:
            run(args, dataset, kind)


if __name__ == '__main__':
    main()

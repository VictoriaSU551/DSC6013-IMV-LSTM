"""Auditable chronological one-step-ahead datasets; no future-value filling."""
import json
import numpy as np
import pandas as pd
from common import ROOT, save_json, sha256


def make_windows(raw, target, depth, segments=None):
    raw = np.asarray(raw, dtype=np.float64)
    idx = np.arange(depth, len(raw))
    x = np.stack([raw[i-depth:i] for i in idx])
    y = raw[idx, target]
    valid = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y)
    if segments is not None:
        segments = np.asarray(segments)
        valid &= np.array([np.all(segments[i-depth:i+1] == segments[i]) for i in idx])
    return x[valid], y[valid], idx[valid]


def scale_splits(x, y, idx, cuts):
    masks = {'train': idx < cuts[0], 'val': (idx >= cuts[0]) & (idx < cuts[1]), 'test': idx >= cuts[1]}
    if not all(m.any() for m in masks.values()):
        raise ValueError('Empty split')
    train_x, train_y = x[masks['train']], y[masks['train']]
    xmin, xmax = train_x.min((0, 1)), train_x.max((0, 1))
    xscale = np.where(xmax > xmin, xmax - xmin, 1.)
    ymin, ymax = train_y.min(), train_y.max()
    yscale = float(ymax-ymin) if ymax > ymin else 1.
    d = {'x_min': xmin, 'x_scale': xscale, 'y_min': np.array(ymin), 'y_scale': np.array(yscale)}
    for name, mask in masks.items():
        d[name+'_x'] = ((x[mask]-xmin)/xscale).astype(np.float32)
        d[name+'_y'] = ((y[mask]-ymin)/yscale).astype(np.float32)
        d[name+'_idx'] = idx[mask]
        d[name+'_raw_y'] = y[mask]
    return d


def read_raw(name):
    folder = ROOT / 'data/raw'
    segments = None
    if name == 'sml':
        files = [folder/'SML2010'/f'NEW-DATA-{i}.T15.txt' for i in [1, 2]]
        parts = []
        for file in files:
            cols = file.read_text().splitlines()[0].split()[1:]
            frame = pd.read_csv(file, sep=r'\s+', skiprows=1, names=cols)
            stamp = pd.to_datetime(frame['1:Date']+' '+frame['2:Time'], dayfirst=True)
            frame['segment'] = (stamp.diff() != pd.Timedelta(minutes=15)).cumsum() + len(parts)*10000
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True)
        # Paper Figure 3 / Table 6: 16 exogenous series + room temperature history.
        positions = [3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 22, 23, 4]
        features = [next(c for c in frame if c.startswith(str(i)+':')) for i in positions]
        target = features[-1]
        segments = frame['segment'].to_numpy()
        cuts = (3200, 3600)
    elif name == 'pm25':
        files = [folder/'PM25/PRSA_data_2010.1.1-2014.12.31.csv']
        frame = pd.read_csv(files[0])
        features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
        for label in ['NE', 'NW', 'SE', 'cv']:
            col = 'wind_'+label
            frame[col] = (frame['cbwd'] == label).astype(float)
            frame.loc[~frame['cbwd'].isin(['NE', 'NW', 'SE', 'cv']), col] = np.nan
            features.append(col)
        features.append('pm2.5')
        target = 'pm2.5'
        cuts = (int(len(frame)*.6), int(len(frame)*.8))
    elif name == 'nasdaq':
        files = [folder/'NASDAQ/nasdaq100_padding.csv']
        frame = pd.read_csv(files[0])
        features, target, cuts = list(frame.columns), 'NDX', (35100, 37830)
    else:
        raise ValueError(name)
    return frame[features].to_numpy(float), features, features.index(target), cuts, segments, files


def main():
    lock = json.loads((ROOT/'sources.lock.json').read_text(encoding='utf-8'))
    for asset in lock['assets']:
        if asset['kind'] == 'data' and sha256(ROOT/asset['path']) != asset['sha256']:
            raise ValueError('Raw data hash differs: '+asset['path'])
    for name in ['sml', 'pm25', 'nasdaq']:
        raw, features, target, cuts, segments, files = read_raw(name)
        x, y, idx = make_windows(raw, target, 10, segments)
        data = scale_splits(x, y, idx, cuts)
        folder = ROOT/'data/processed'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/(name+'.npz'), **data)
        info = {'dataset': name, 'protocol': 'chronological-v1', 'raw_rows': len(raw), 'features': features,
                'target': features[target], 'target_column': target, 'depth': 10, 'horizon': 1,
                'cuts': list(cuts), 'invalid_or_segment_crossing_windows': len(raw)-10-len(idx),
                'splits': {s: {'n': len(data[s+'_y']), 'first_target': int(data[s+'_idx'][0]),
                               'last_target': int(data[s+'_idx'][-1])} for s in ['train', 'val', 'test']},
                'raw_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
                'npz_sha256': sha256(folder/(name+'.npz')),
                'preparation_code_sha256': sha256(__file__)}
        save_json(folder/(name+'.json'), info)
        print(name, {s: v['n'] for s, v in info['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()

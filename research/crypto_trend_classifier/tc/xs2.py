"""Follow-up to xs.py: same comparison on BIG trends (oracle k=2, ~2x the switching cost), where regime /
crash information should matter most. Same frozen params for both models; only the columns differ."""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dataset import train_rows
from .tune import TRAIN_ASSETS
from .labels import oracle_labels
from .stage2 import params
from .evaluate import smooth_state, metrics
from .xs import fr_fn, XS, EVAL

K = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
CONTROL = len(sys.argv) > 2 and sys.argv[2] == 'seed'  # noise control: same columns, different seed


def gather_k(cutoff):
    parts = [train_rows(fr_fn(a, tf), pd.Timestamp(cutoff, tz='UTC'), k=K) for a in TRAIN_ASSETS for tf in ('1h', '4h')]
    return pd.concat([p for p in parts if len(p)])


def main():
    fz = json.load(open('FROZEN_v3.json'))['C']
    p = fz['params']
    models = {'C (frozen cols)': fz['cols'], 'C + cross-section': fz['cols'] + XS}
    if CONTROL:
        models = {'C (frozen cols)': fz['cols'], 'C + cross-section': list(fz['cols'])}  # 2nd = same cols, seed 1
    cuts = ['2019-01-01', '2021-01-01', '2023-01-01', '2025-01-01', '2026-09-24']
    labs = {(a, tf): oracle_labels(fr_fn(a, tf).close.values, K) for a in EVAL for tf in ('1h', '4h')}
    rows = []
    for i in range(4):
        tr = gather_k(cuts[i])
        lo, hi = pd.Timestamp(cuts[i], tz='UTC'), pd.Timestamp(cuts[i + 1], tz='UTC')
        assert tr.index.max() < lo
        for mn, cols in models.items():
            prm = params(p)
            if CONTROL and mn != 'C (frozen cols)':
                prm['seed'] = 1
            m = lgb.train(prm, lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
            for a in EVAL:
                for tf in ('1h', '4h'):
                    fr = fr_fn(a, tf)
                    w = (fr.index >= lo) & (fr.index < hi)
                    te = fr[w]
                    s = smooth_state(m.predict(te[cols]), p['smooth_span'], p['hysteresis'])
                    mt = metrics(s, labs[(a, tf)][w], te.close.values)
                    mt.update(model=mn, asset=a, tf=tf, fold=cuts[i])
                    rows.append(mt)
        print('fold', cuts[i], flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle(f'out_xs_k{K}{"_seed" if CONTROL else ""}.pkl')
    k = ['mcc', 'flip_precision', 'flips_per_oracle_flip', 'missed_move_frac', 'capture']
    pd.set_option('display.width', 200)
    print(R.groupby(['tf', 'model'])[k].mean().round(3).to_string())
    w = R.pivot_table(index=['asset', 'tf', 'fold'], columns='model', values='mcc')
    print('cells where cross-section improves MCC:', int((w['C + cross-section'] > w['C (frozen cols)']).sum()), 'of', len(w))


if __name__ == '__main__':
    main()

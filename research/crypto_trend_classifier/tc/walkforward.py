"""
Step 2 - anchored walk-forward, retrained every year with the FROZEN settings from tune.py.
For test year Y the model sees only rows whose features are complete AND whose oracle label is already final
at Y-01-01 (labels recomputed on truncated data), minus an embargo. It then predicts every bar of year Y.
Out-of-sample probabilities of all years are stitched per series; smoothing is causal, so the state at the
start of a year only uses the previous (also out-of-sample) predictions.

Variants:
  full      - all coins, 1h + 4h training, BTC context features
  btc_only  - trained on BTC only (Bitstamp 2013-17 + Binance), applied to every coin  -> unseen-coin test
  no_ctx    - all coins but without the BTC context features
2h candles are never used in training -> unseen-timeframe test.
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dataset import frame, COINS
from .tune import gather, feat_cols, ASSETS, TRAIN_ASSETS

YEARS = list(range(2019, 2027))
PLACEBO_YEARS = [2023, 2024, 2025, 2026]
EVAL_TFS = ['1h', '2h', '4h']


def run(variant):
    cfg = json.load(open('out_tc_tuning.json'))['model']
    prm = dict(objective='binary', verbosity=-1, num_threads=4, seed=0, bagging_freq=1,
               **{k: v for k, v in cfg.items() if k not in ('n_estimators', 'smooth_span', 'hysteresis')})
    n_est = cfg['n_estimators']
    train_assets = ['BTC_BITSTAMP', 'BTCUSDT', 'BTC_BITSTAMP_INV', 'BTCUSDT_INV'] if variant == 'btc_only' else TRAIN_ASSETS
    out = []
    for Y in (PLACEBO_YEARS if variant == 'placebo' else YEARS):
        cut = f'{Y}-01-01'
        tr = gather(cut, assets=train_assets)
        if variant == 'placebo':
            # PLACEBO: break the link between features and labels by rolling each series' labels half its length.
            # A leak-free pipeline must then score ~0 out-of-sample.
            tr = tr.copy()
            for _, idx in tr.groupby(['asset', 'tf'], sort=False).indices.items():
                tr.iloc[idx, tr.columns.get_loc('y')] = np.roll(tr.y.values[idx], len(idx) // 2)
        cols = feat_cols(tr)
        if variant == 'no_ctx':
            cols = [c for c in cols if not c.startswith(('btc_', 'rel_'))]
        m = lgb.train(prm, lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), n_est)
        lo, hi = pd.Timestamp(cut, tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        for a in COINS:
            for tf in EVAL_TFS:
                fr = frame(a, tf)
                te = fr[(fr.index >= lo) & (fr.index < hi)]
                if not len(te):
                    continue
                assert te.index.min() >= lo and tr.index.max() < lo  # no training row inside the test year
                p = m.predict(te[cols])
                out.append(pd.DataFrame(dict(asset=a, tf=tf, p=p, y=te.y.values, final=te.final.values,
                                             close=te.close.values, year=Y), index=te.index))
        print(variant, Y, 'train rows', len(tr), 'last train bar', tr.index.max(), flush=True)
        if variant == 'full' and Y == 2026:
            imp = pd.Series(m.feature_importance('gain'), cols).sort_values(ascending=False)
            imp.to_csv('out_tc_importance.csv')
    pd.concat(out).to_pickle(f'out_tc_oos_{variant}.pkl')


if __name__ == '__main__':
    run(sys.argv[1])

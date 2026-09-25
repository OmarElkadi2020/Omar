"""Train the production model on every label that is final today (v2 settings, frozen pre-2019) and print the
current trend state of each coin. Re-run after refreshing the data to update."""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .v2 import frame2
from .tune import gather, feat_cols
from .dataset import COINS
from .evaluate import smooth_state


def train(cutoff):
    cfg = json.load(open('out_tc_tuning_v2.json'))['model']
    prm = dict(objective='binary', verbosity=-1, num_threads=4, seed=0, bagging_freq=1,
               **{k: v for k, v in cfg.items() if k not in ('n_estimators', 'smooth_span', 'hysteresis')})
    tr = gather(cutoff, fr_fn=frame2)
    cols = feat_cols(tr)
    m = lgb.train(prm, lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), cfg['n_estimators'])
    m.save_model('trend_model_v2.txt')
    json.dump(dict(features=cols, smooth_span=cfg['smooth_span'], hysteresis=cfg['hysteresis'], trained_until=cutoff,
                   train_rows=len(tr)), open('trend_model_v2.json', 'w'), indent=1)
    return m, cols, cfg


def now(m, cols, cfg):
    rows = []
    for a in COINS:
        for tf in ['1h', '4h']:
            fr = frame2(a, tf)
            p = m.predict(fr[cols].iloc[-3000:])
            s = smooth_state(p, cfg['smooth_span'], cfg['hysteresis'])
            flips = np.flatnonzero(s[1:] != s[:-1])
            since = len(s) - 1 - flips[-1] if len(flips) else len(s)
            rows.append(dict(coin=a, tf=tf, last_bar=fr.index[-1], trend='UP' if s[-1] == 1 else 'DOWN',
                             p_up=round(float(p[-1]), 3), bars_in_state=int(since)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    cut = sys.argv[1] if len(sys.argv) > 1 else '2026-09-24'
    m, cols, cfg = train(cut)
    print(now(m, cols, cfg).to_string(index=False))

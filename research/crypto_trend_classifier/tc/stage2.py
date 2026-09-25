"""Stage 2: ONE walk-forward run 2019-2026 of the configuration frozen in FROZEN_v3.json (see PREREG_stage1.md)."""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dev3 import frame3
from .dataset import COINS
from .tune import gather


def params(p):
    return dict(objective='binary', verbosity=-1, num_threads=4, seed=0, bagging_freq=1,
                **{k: v for k, v in p.items() if k not in ('n_estimators', 'smooth_span', 'hysteresis')})


def run(tag):
    fz = json.load(open('FROZEN_v3.json'))[tag]
    cols, p = fz['cols'], fz['params']
    out = []
    for Y in range(2019, 2027):
        cut = f'{Y}-01-01'
        tr = gather(cut, fr_fn=frame3)
        lo, hi = pd.Timestamp(cut, tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        assert tr.index.max() < lo
        m = lgb.train(params(p), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
        for a in COINS:
            for tf in ['1h', '2h', '4h']:
                fr = frame3(a, tf)
                te = fr[(fr.index >= lo) & (fr.index < hi)]
                if len(te):
                    out.append(pd.DataFrame(dict(asset=a, tf=tf, p=m.predict(te[cols]), y=te.y.values,
                                                 final=te.final.values, close=te.close.values, year=Y), index=te.index))
        print(tag, Y, len(tr), flush=True)
    pd.concat(out).to_pickle(f'out_stage2_{tag}.pkl')


if __name__ == '__main__':
    run(sys.argv[1])

"""dev only: what a veto of 'probably false' down-flips would do (stay long 30 days instead of exiting)."""
import numpy as np, pandas as pd
from . import stage27 as s27
from .diag28 import DEV_LO, DEV_HI
E = pd.read_pickle('cache_stage16/s28_events.pkl')
E = E[(E.t >= DEV_LO) & (E.t < DEV_HI)].copy()
fwd = []
for (c, tf), g in E.groupby(['coin', 'tf']):
    f = s27.bars(c, tf); H = int(30 * 24 / s27.HOURS[tf])
    i = f.index.get_indexer(g.t); cl = f.close.values
    fwd.append(pd.Series(np.log(cl[np.minimum(i + H, len(cl) - 1)] / cl[i]), index=g.index))
E['fwd30'] = pd.concat(fwd)
rows = []
for tf, feat, sign in (('1h', 'chop14', 1), ('1h', 'seg_gain', 1), ('1h', 'rsi14', 1), ('4h', 'chop50', 1), ('4h', 'roc20', 1),
                       ('8h', 'chop50', 1), ('8h', 'roc100', 1)):
    d = E[E.tf == tf].dropna(subset=[feat])
    for q in (0.9, 0.75, 0.5):
        th = d[feat].quantile(q)
        v = d[d[feat] * sign >= th * sign]; k = d[d[feat] * sign < th * sign]
        rows.append(dict(tf=tf, veto_if=f'{feat} in top {int(round((1 - q) * 100))}%', n_veto=len(v), false_share_vetoed=v.false.mean(),
                         false_share_kept=k.false.mean(), base_false=d.false.mean(),
                         mean_fwd30_vetoed=v.fwd30.mean(), median_fwd30_vetoed=v.fwd30.median(), worst10pct_vetoed=v.fwd30.quantile(0.1)))
R = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False)); R.to_csv('results_diag28_dev_veto.csv', index=False)

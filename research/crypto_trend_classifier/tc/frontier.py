"""Accuracy-vs-whipsaw frontier, out-of-sample 2019-2026. The model's operating point used in the headline is
the frozen pre-2019 one; the curves here only DESCRIBE the trade-off, nothing is selected on them."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from .dataset import frame, COINS
from .evaluate import smooth_state, BASELINE_GRID, BASELINE_FN


def score(states, oos):
    m, f = [], []
    for a, (s, o) in states.items():
        k = o.final.values
        y = o.y.values[k]
        s = s[k]
        m.append(matthews_corrcoef(y, s))
        f.append(np.sum(s[1:] != s[:-1]) / max(np.sum(y[1:] != y[:-1]), 1))
    return np.mean(m), np.mean(f)


def main(variant, tf):
    oos = pd.read_pickle(f'out_tc_oos_{variant}.pkl')
    oos = oos[oos.tf == tf]
    per = {a: oos[oos.asset == a] for a in COINS}
    rows = []
    for span in (1, 2, 4, 8, 16, 32, 64, 128):
        for h in (0, .05, .1, .15, .2, .25, .3, .35, .4):
            st = {a: (smooth_state(o.p.values, span, h), o) for a, o in per.items()}
            mc, fl = score(st, per)
            rows.append(dict(method='Model', params=f'span={span},h={h}', mcc=mc, flips=fl))
    for name, grid in BASELINE_GRID.items():
        frs = {a: frame(a, tf) for a in COINS}
        for prm in grid:
            st = {a: (pd.Series(BASELINE_FN[name](frs[a], prm), frs[a].index).reindex(o.index).values, o) for a, o in per.items()}
            mc, fl = score(st, per)
            rows.append(dict(method=name, params=str(prm), mcc=mc, flips=fl))
    pd.DataFrame(rows).to_pickle(f'out_tc_frontier_{variant}_{tf}.pkl')
    print(pd.DataFrame(rows).sort_values('mcc', ascending=False).groupby('method').head(3).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from .dataset import frame, COINS
from .evaluate import metrics, smooth_state, BASELINE_FN, BASELINE_GRID
from .report import states_for_series, BPY


def main():
    fz = json.load(open('FROZEN_v3.json'))['C']['params']
    t1 = json.load(open('out_tc_tuning.json'))
    t2 = json.load(open('out_tc_tuning_v2.json'))['model']
    o3 = pd.read_pickle('out_stage2_C.pkl')
    o2 = pd.read_pickle('out_tc_oos_v2.pkl')
    rows, front = [], []
    for tf in ['1h', '2h', '4h']:
        for a in COINS:
            fr = frame(a, tf)
            x = o3[(o3.asset == a) & (o3.tf == tf)]
            x2 = o2[(o2.asset == a) & (o2.tf == tf)].reindex(x.index)
            S = states_for_series(x, x.p.values, fz, t1['baselines'], tf, fr)
            S['Model v3 (frozen C)'] = S.pop('Model (LightGBM, walk-forward)')
            S['Model v2 (previous)'] = smooth_state(x2.p.values, t2['smooth_span'], t2['hysteresis'])
            k = x.final.values
            y = x.y.values[k]
            for n, s in S.items():
                m = metrics(s[k], y, x.close.values[k], bars_per_year=BPY[tf])
                m.update(asset=a, tf=tf, clf=n)
                rows.append(m)
            fl = lambda s: np.sum(s[1:] != s[:-1]) / max(np.sum(y[1:] != y[:-1]), 1)
            for span in (1, 2, 4, 8, 16, 32, 64):
                for h in (0, .1, .2, .3, .4):
                    s = smooth_state(x.p.values, span, h)[k]
                    front.append(dict(tf=tf, asset=a, method='Model v3', params=f'{span},{h}', mcc=matthews_corrcoef(y, s), flips=fl(s)))
            for bn, grid in BASELINE_GRID.items():
                for prm in grid:
                    s = pd.Series(BASELINE_FN[bn](fr, prm), fr.index).reindex(x.index).values[k]
                    front.append(dict(tf=tf, asset=a, method=bn, params=str(prm), mcc=matthews_corrcoef(y, s), flips=fl(s)))
    R = pd.DataFrame(rows)
    F = pd.DataFrame(front)
    R.to_pickle('out_stage2_rows.pkl')
    F.to_pickle('out_stage2_frontier.pkl')
    return R, F


if __name__ == '__main__':
    R, F = main()
    pd.set_option('display.width', 250)
    cols = ['acc', 'mcc', 'flips_per_oracle_flip', 'flip_precision', 'missed_move_frac', 'exit_giveback_frac', 'ls_sharpe', 'lf_sharpe']
    print(R.groupby(['tf', 'clf'])[cols].mean().round(3).to_string())
    # frontier: average per (method, params) over coins, then best under a flip budget
    F['params'] = F['params'].fillna('')
    for tf in ['1h', '2h', '4h']:
        g = F[F.tf == tf]
        g = g.assign(key=g.method + g.params.astype(str)).groupby(['method', 'key'])[['mcc', 'flips']].mean().reset_index()
        g.loc[g.method == 'Model v3', 'key'] = g.loc[g.method == 'Model v3', 'key']
        for b in (1.5, 2, 3):
            print(tf, b, {m: round(d[d.flips <= b].mcc.max(), 3) for m, d in g.groupby('method')})

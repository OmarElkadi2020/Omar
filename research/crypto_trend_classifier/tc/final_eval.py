"""Final out-of-sample table 2019-2026: v2 model at its FROZEN pre-2019 operating point vs everything else."""
import json
import numpy as np
import pandas as pd
from .dataset import frame, COINS
from .evaluate import metrics, smooth_state
from .report import states_for_series, BPY


def main():
    t1 = json.load(open('out_tc_tuning.json'))
    t2 = json.load(open('out_tc_tuning_v2.json'))
    o1 = pd.read_pickle('out_tc_oos_full.pkl')
    o2 = pd.read_pickle('out_tc_oos_v2.pkl')
    ob = pd.read_pickle('out_tc_oos_btc_only.pkl')
    rows = []
    for tf in ['1h', '2h', '4h']:
        for a in COINS:
            fr = frame(a, tf)
            x2 = o2[(o2.asset == a) & (o2.tf == tf)]
            x1 = o1[(o1.asset == a) & (o1.tf == tf)].reindex(x2.index)
            xb = ob[(ob.asset == a) & (ob.tf == tf)].reindex(x2.index)
            S = states_for_series(x2, x2.p.values, t2['model'], t1['baselines'], tf, fr)
            S['Model v2 (breadth + whipsaw cap)'] = S.pop('Model (LightGBM, walk-forward)')
            S['Model v1 (no whipsaw cap)'] = smooth_state(x1.p.values, t1['model']['smooth_span'], t1['model']['hysteresis'])
            S['Model v2 trained on BTC only'] = smooth_state(xb.p.values, t2['model']['smooth_span'], t2['model']['hysteresis'])
            k = x2.final.values
            for n, s in S.items():
                m = metrics(s[k], x2.y.values[k], x2.close.values[k], bars_per_year=BPY[tf])
                m.update(asset=a, tf=tf, clf=n)
                rows.append(m)
    R = pd.DataFrame(rows)
    R.to_pickle('out_tc_final_rows.pkl')
    return R


if __name__ == '__main__':
    R = main()
    pd.set_option('display.width', 260)
    cols = ['acc', 'mcc', 'flips_per_oracle_flip', 'flip_precision', 'missed_move_frac', 'exit_giveback_frac', 'capture', 'ls_sharpe', 'lf_sharpe', 'bh_sharpe']
    print(R.groupby(['tf', 'clf'])[cols].mean().round(3).to_string())

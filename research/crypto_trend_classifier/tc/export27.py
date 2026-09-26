"""BTC candles + SuperTrend lines (frozen stage-27 choices) for the chart artifact."""
import json
import numpy as np
import pandas as pd
from .features import _atr
from . import stage27 as s

VIEW = {'1d': '2800D', '4h': '365D', '1h': '90D', '15m': '21D', '5m': '7D'}


def st_line(f, p, m):
    a = _atr(f, p).values
    h, l, c = f.high.values, f.low.values, f.close.values
    n = len(c)
    line, dirn = np.full(n, np.nan), np.zeros(n, int)
    fub = flb = np.nan
    dd = 1
    for i in range(n):
        if np.isnan(a[i]):
            continue
        hl2 = (h[i] + l[i]) / 2
        ub, lb = hl2 + m * a[i], hl2 - m * a[i]
        if np.isnan(fub):
            fub, flb = ub, lb
        else:
            fub = ub if (ub < fub or c[i - 1] > fub) else fub
            flb = lb if (lb > flb or c[i - 1] < flb) else flb
        if dd == 1 and c[i] < flb:
            dd = -1
        elif dd == -1 and c[i] > fub:
            dd = 1
        line[i] = flb if dd == 1 else fub
        dirn[i] = dd
    return line, dirn


def main():
    fz = json.load(open('FROZEN_stage27.json'))['choices']
    T = pd.read_csv('results_stage27_test.csv')
    out = {}
    for tf in VIEW:
        f = s.bars('BTC', tf)
        # check against the tested implementation
        ref = s.s26.st_state(f, 10, 3.0)
        chk = st_line(f, 10, 3.0)[1]
        ok = np.isfinite(st_line(f, 10, 3.0)[0])
        assert ((chk[ok] == 1) == (ref[ok] == 1)).all()
        k = f.index >= f.index[-1] - pd.Timedelta(VIEW[tf])
        g = f[k]
        d = dict(t=g.index.tz_convert(None).values.astype('datetime64[s]').astype('int64').tolist(),
                 o=g.open.round(2).tolist(), h=g.high.round(2).tolist(), l=g.low.round(2).tolist(), c=g.close.round(2).tolist(),
                 settings={})
        for key, name in (('acc_choice', 'trend accuracy'), ('filter_choice', 'long filter')):
            p, m = fz[tf][key]
            line, dirn = st_line(f, int(p), float(m))
            row = T[(T.tf == tf) & (T.setting == ('ACC choice' if key == 'acc_choice' else 'FILTER choice'))].iloc[0]
            d['settings'][key] = dict(name=name, p=int(p), m=float(m), line=np.round(line[k], 2).tolist(), dir=dirn[k].tolist(),
                                      flips=int(np.abs(np.diff(dirn[k])).sum() // 2),
                                      test=dict(mcc=round(float(row.mcc_med), 3), sharpe=round(float(row.sharpe), 2),
                                                delay_h=round(float(row.delay_hours), 1), fpt=round(float(row.fpt_med), 2),
                                                acc_pct=round(float(row.acc_pct), 2), sharpe_pct=round(float(row.sharpe_pct), 2),
                                                maxdd=round(float(row.maxdd), 3)))
        out[tf] = d
        print(tf, len(g), d['t'][0], {k2: (v['p'], v['m'], v['flips']) for k2, v in d['settings'].items()})
    json.dump(out, open('art/btc_st_data.json', 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()

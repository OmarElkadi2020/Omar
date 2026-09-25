"""Stage 7: the trend indicator as a standalone trading tool (see PREREG_stage7_trading.md). Run once."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dev3 import frame3
from .stage3 import frame_generic, FX, IDX, STOCKS
from .evaluate import smooth_state, BASELINE_FN, price_ema

EVAL = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT']
COST = {'Crypto': 0.001, 'FX': 0.0001, 'Indices': 0.0002, 'Stocks': 0.0005}


def perf(pos, c, cost, bpy):
    r = np.r_[np.diff(np.log(c)), 0.0]
    pnl = pos * r - cost * np.abs(np.diff(pos, prepend=0.0))
    eq = np.cumsum(pnl)
    yrs = len(c) / bpy
    dd = np.exp(eq - np.maximum.accumulate(np.maximum(eq, 0))) - 1
    cagr = np.exp(eq[-1] / yrs) - 1
    mdd = dd.min()
    return dict(sharpe=pnl.mean() / pnl.std() * np.sqrt(bpy) if pnl.std() > 0 else 0.0, cagr=cagr, max_dd=mdd,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan, in_mkt=np.mean(pos != 0),
                trips_per_yr=np.sum((pos[1:] > 0) & (pos[:-1] <= 0)) / yrs)


def signals(fr, p, tun, sl):
    S = {'MODEL': smooth_state(p, 1, 0.204)}
    for bn in BASELINE_FN:
        S[bn] = np.asarray(BASELINE_FN[bn](fr, tun[bn]['params']))[sl]
    S['EMA200'] = price_ema(fr, 200)[sl]
    S['EMA100'] = price_ema(fr, 100)[sl]
    return S


def series():
    fz = json.load(open('FROZEN_v3.json'))['C']
    tun = json.load(open('out_tc_tuning.json'))['baselines']
    o = pd.read_pickle('out_stage2_C.pkl')
    for a in EVAL:
        for tf in ('1h', '4h'):
            fr = frame3(a, tf)
            x = o[(o.asset == a) & (o.tf == tf)]
            sl = fr.index.get_indexer(x.index)
            yield 'Crypto', a, tf, fr, x.p.values, sl, tun[tf], {'1h': 8760, '4h': 2190}[tf]
    m = lgb.Booster(model_file='trend_model_universal_C.txt')
    import os
    for grp, names, tfs in (('FX', FX, ('1h', '4h')), ('Indices', IDX, ('1h', '4h')), ('Stocks', STOCKS, ('1D',))):
        for n in names:
            for tf in tfs:
                if not os.path.exists(f'cache_stage3/{n}_{tf}.pkl'):
                    continue
                fr = frame_generic(n, tf)
                if len(fr) < 2000:
                    continue
                sl = np.arange(1000, len(fr))
                p = m.predict(fr[fz['cols']])[sl]
                days = (fr.index[-1] - fr.index[1000]).days
                yield grp, n, tf, fr, p, sl, tun['4h' if tf == '1D' else tf], len(sl) / (days / 365.25)


def main():
    rows = []
    for grp, a, tf, fr, p, sl, tun, bpy in series():
        c = fr.close.values[sl]
        S = signals(fr, p, tun, sl)
        for name, s in S.items():
            s = np.asarray(s, float)
            for mode in ('long-flat', 'long-short'):
                pos = np.maximum(s, 0) if mode == 'long-flat' else s
                for rob, pp, k in (('base', pos, 1), ('delay+1', np.r_[0, pos[:-1]], 1), ('cost x2', pos, 2)):
                    rows.append(dict(group=grp, asset=a, tf=tf, signal=name, mode=mode, robust=rob,
                                     **perf(pp, c, COST[grp] * k, bpy)))
        rows.append(dict(group=grp, asset=a, tf=tf, signal='Buy & hold', mode='long-flat', robust='base',
                         **perf(np.ones(len(c)), c, 0.0, bpy)))
        print(grp, a, tf, len(c), fr.index[sl[0]].date(), fr.index[sl[-1]].date(), flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage7.pkl')


if __name__ == '__main__':
    main()

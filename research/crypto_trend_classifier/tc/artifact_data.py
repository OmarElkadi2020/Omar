"""Data for the Lightweight-Charts artifact: top-5 coins, 1D and 4h, 2019-01-01 .. end, trend states + stats."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from .stage11 import basket_market, frame
from .evaluate import smooth_state

COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
COST = 0.001
BPY = {'1D': 365, '4h': 2190}
T0 = pd.Timestamp('2019-01-01', tz='UTC')


def sig(x):
    return float(f'{x:.6g}')


def stats(pos, c, y, bpy):
    r = np.r_[np.diff(np.log(c)), 0.0]
    pnl = pos * r - COST * np.abs(np.diff(pos, prepend=0.0))
    eq = np.cumsum(pnl)
    dd = np.exp(eq - np.maximum.accumulate(np.maximum(eq, 0))) - 1
    yrs = len(c) / bpy
    s = np.where(pos > 0, 1, -1)
    fl = np.flatnonzero(s[1:] != s[:-1]) + 1
    return dict(sharpe=pnl.mean() / pnl.std() * np.sqrt(bpy), cagr=np.exp(eq[-1] / yrs) - 1, maxdd=dd.min(),
                total=np.exp(eq[-1]) - 1, trades=int(np.sum((pos[1:] > 0) & (pos[:-1] <= 0)) + (pos[0] > 0)),
                inmkt=float(pos.mean()), mcc=matthews_corrcoef(y, s),
                flips=len(fl) / max(np.sum(y[1:] != y[:-1]), 1), flip_prec=float(np.mean(y[fl] == s[fl])) if len(fl) else None)


def main():
    fz, fzb = json.load(open('FROZEN_stage10.json')), json.load(open('FROZEN_stage10b.json'))
    ms = lgb.Booster(model_file='trend_model_stocks.txt')
    o = pd.read_pickle('out_stage2_C.pkl')
    out = {}
    for tf in ('1D', '4h'):
        m = basket_market(tf)
        for a in COINS:
            f = frame(a, tf, m)
            pr = ms.predict(f[fz['cols']])
            st = {'acc': smooth_state(pr, fz['params']['smooth_span'], fz['params']['hysteresis']),
                  'calm': smooth_state(pr, fzb['span'], fzb['h'])}
            if tf == '4h':
                x = o[(o.asset == a) & (o.tf == tf)]
                cm = pd.Series(np.nan, f.index)
                ii = x.index.intersection(f.index)
                cm.loc[ii] = x.p.reindex(ii).values
                st['c'] = smooth_state(cm.values, 1, 0.204)
            w = f.index >= T0
            g = f[w]
            c = g.close.values
            y = g.y.values.astype(int)
            d = dict(t=[int(t.timestamp()) for t in g.index],
                     o=[sig(v) for v in g.open.values], h=[sig(v) for v in g.high.values],
                     l=[sig(v) for v in g.low.values], c=[sig(v) for v in c],
                     ideal=''.join('+' if v == 1 else '-' for v in y), states={}, stats={})
            for k, s in st.items():
                s = np.asarray(s)[w]
                d['states'][k] = ''.join('+' if v == 1 else '-' for v in s)
                d['stats'][k] = stats((s == 1).astype(float), c, y, BPY[tf])
            d['stats']['bh'] = stats(np.ones(len(c)), c, y, BPY[tf])
            out[f'{a}|{tf}'] = d
            print(tf, a, len(g), {k: round(v['sharpe'], 2) for k, v in d['stats'].items()}, flush=True)
    s = json.dumps(out, separators=(',', ':'), default=float)
    open('artifact_data.json', 'w').write(s)
    print('bytes', len(s))


if __name__ == '__main__':
    main()

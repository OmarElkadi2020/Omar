"""Stage 14: stage-12 model as early entry + original ATR exits (see PREREG_stage14_hybrid.md)."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import build, build_bb, _atr
from .labels import oracle_labels
from .dataset import load_1h
from .stage3 import load as oanda_load, FX, IDX
from .stage10 import chop_features
from .s12lib import new_features
from .stage12 import score as s12score, to_state as s12state, parse_cfg, fr as fr12, DEV1
from .stage10 import splits
from .strat_lag import adx
from .evaluate import supertrend

COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT', 'DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']
COST = {'Crypto 4h': 0.001, 'Crypto 1D': 0.001, 'Gold+FX': 0.0001, 'Indices': 0.0002, 'Stocks': 0.0005}
BPY = {'4h': None, '1D': None}


def ohlcv(loader, name, tf):
    d = loader(name)
    return d.resample(tf, origin='epoch').agg(dict(open='first', high='max', low='min', close='last',
                                                   volume='sum')).dropna(subset=['close'])


def market(closes):
    P = pd.DataFrame(closes).sort_index()
    R = np.log(P).diff().clip(-0.4, 0.4)
    idx = np.exp(R.mean(axis=1).fillna(0).cumsum())
    m = pd.DataFrame(index=P.index)
    for n in (21, 63, 252):
        m[f'mkt_mom_{n}'] = np.log(idx / idx.shift(n))
    m['mkt_ema_dist_200'] = np.log(idx / idx.ewm(span=200, adjust=False).mean())
    s200 = P.rolling(200, min_periods=200).mean()
    m['mkt_breadth_200'] = (P > s200).sum(axis=1) / s200.notna().sum(axis=1).replace(0, np.nan)
    return m


def frame(loader, name, tf, m):
    d = loader(name)
    base, X = build(d, tf)
    X = pd.concat([X, build_bb(d, tf).reindex(X.index)], axis=1)
    mk = m.reindex(X.index)
    X = pd.concat([X, mk], axis=1)
    c = base.close
    for n in (63, 252):
        X[f'rel_mom_{n}'] = np.log(c / c.shift(n)) - mk[f'mkt_mom_{n}']
    X = pd.concat([X, chop_features(base)], axis=1)
    g = ohlcv(loader, name, tf).reindex(X.index)
    X = pd.concat([X, new_features(g.open, g.high, g.low, g.close, g.volume)], axis=1)
    f = pd.concat([base[['open', 'high', 'low', 'close']], X.astype(np.float32)], axis=1)
    return f.loc[:, ~f.columns.duplicated()]


def simulate(f, long_sig, short_sig, cost):
    """ATR exits of the original strategy; signals are evaluated at the bar close."""
    c = f.close.values
    atr = _atr(f, 14).values
    n = len(c)
    pos = np.zeros(n)
    cur, stop, best, tp = 0, 0.0, 0.0, None
    trades, wins, entry = 0, 0, 0.0
    for i in range(n):
        if cur == 1:
            best = max(best, c[i])
            stop = max(stop, best - 6.5 * atr[i])
            if c[i] <= stop:
                wins += c[i] > entry
                cur = 0
        elif cur == -1:
            best = min(best, c[i])
            stop = min(stop, best + 3.5 * atr[i])
            if c[i] >= stop or c[i] <= tp:
                wins += c[i] < entry
                cur = 0
        if cur == 0 and atr[i] > 0:
            if long_sig[i]:
                cur, stop, best, entry = 1, c[i] - 3 * atr[i], c[i], c[i]
                trades += 1
            elif short_sig[i]:
                cur, stop, best, tp, entry = -1, c[i] + 4 * atr[i], c[i], c[i] - 5 * atr[i], c[i]
                trades += 1
        pos[i] = cur
    return pos, trades, wins


def perf(pos, c, cost, bpy):
    r = np.r_[np.diff(np.log(c)), 0.0]
    pnl = pos * r - cost * np.abs(np.diff(pos, prepend=0.0))
    eq = np.cumsum(pnl)
    yrs = len(c) / bpy
    dd = np.exp(eq - np.maximum.accumulate(np.maximum(eq, 0))) - 1
    return dict(sharpe=pnl.mean() / pnl.std() * np.sqrt(bpy) if pnl.std() > 0 else 0.0,
                cagr=np.exp(eq[-1] / yrs) - 1, max_dd=dd.min(), total=np.exp(eq[-1]) - 1, in_mkt=float(np.mean(pos != 0)))


def run_series(group, name, f, state, lo, hi, don, tfbars):
    w = (f.index >= lo) & (f.index < hi)
    c = f.close.values
    st = np.asarray(supertrend(f, 24, 4.0))
    e10 = f.close.ewm(span=10, adjust=False).mean().values
    reg = np.where((st == 1) & (c >= e10), 1, np.where((st == -1) & (c <= e10), -1, 0))
    hi_ = f.high.rolling(don).max().shift(1).values
    lo_ = f.low.rolling(don).min().shift(1).values
    ax = adx(f)
    orig_l = (reg == 1) & (ax >= 10) & (c > hi_)
    orig_s = (reg == -1) & (ax >= 15) & (c < lo_)
    s = np.asarray(state)
    prev = np.r_[0, s[:-1]]
    hyb_l = (s == 1) & (prev != 1)
    hyb_s = (s == -1) & (prev != -1)
    idx = np.flatnonzero(w)
    a, b = idx[0], idx[-1] + 1
    sub = f.iloc[a:b]
    cost = COST[group]
    days = (sub.index[-1] - sub.index[0]).days
    bpy = len(sub) / (days / 365.25)
    rows = []
    for var, (L, S) in (('ORIGINAL', (orig_l, orig_s)), ('HYBRID', (hyb_l, hyb_s)),
                        ('HYBRID-REENTRY (exploratory)', (s == 1, s == -1))):
        pos, tr, wn = simulate(sub, L[a:b], S[a:b], cost)
        rows.append(dict(variant=var, trades=tr, win_rate=wn / max(tr, 1), **perf(pos, sub.close.values, cost, bpy)))
    pos = np.where(s[a:b] == 1, 1.0, np.where(s[a:b] == -1, -1.0, 0.0))
    rows.append(dict(variant='MODEL-FLIP', trades=int(np.sum(np.diff(pos) != 0)), win_rate=np.nan,
                     **perf(pos, sub.close.values, cost, bpy)))
    rows.append(dict(variant='BUY & HOLD', trades=0, win_rate=np.nan, **perf(np.ones(b - a), sub.close.values, 0.0, bpy)))
    for r in rows:
        r.update(group=group, asset=name, start=str(sub.index[0].date()), end=str(sub.index[-1].date()))
    return rows


def model_state(f):
    meta = json.load(open('stage12_model_meta.json'))
    m = lgb.Booster(model_file='trend_model_stage12.txt')
    cols = json.load(open('FROZEN_stage12.json'))['cols']
    return s12state(s12score(m, meta['kind'], meta['sd'], f[cols]), parse_cfg(meta['cfg']))


def main():
    rows = []
    T19, T26 = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2026-09-24', tz='UTC')
    # crypto
    for tf, don in (('4h', 21), ('1D', 4)):
        closes = {a: ohlcv(load_1h, a, tf).close for a in COINS}
        mk = market(closes)
        for a in COINS:
            f = frame(load_1h, a, tf, mk)
            rows += run_series(f'Crypto {tf}', a, f, model_state(f), T19, T26, don, tf)
        print('crypto', tf, flush=True)
    # Oanda 4h
    for grp, names in (('Gold+FX', ['XAU_USD'] + FX), ('Indices', IDX)):
        names = [n for n in names if n != 'DAX'] if grp == 'Indices' else names
        for n in names:
            peers = ['XAU_USD'] if n == 'XAU_USD' else [x for x in names if x != 'XAU_USD']
            mk = market({p: ohlcv(oanda_load, p, '4h').close for p in peers})
            f = frame(oanda_load, n, '4h', mk)
            rows += run_series(grp, n, f, model_state(f), f.index[1000], f.index[-1] + pd.Timedelta('4h'), 21, '4h')
        print(grp, flush=True)
    # stocks (stage-12 cache, daily, set A)
    for t in splits()['A']:
        f = fr12(t)
        rows += run_series('Stocks', t, f, model_state(f), DEV1, T26, 14, '1D')
    print('stocks', flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage14_explore.pkl')
    pd.set_option('display.width', 220)
    print(R.groupby(['group', 'variant'])[['sharpe', 'cagr', 'max_dd', 'trades', 'win_rate', 'in_mkt']].median().round(3).to_string())
    from scipy.stats import binomtest
    w = R.pivot_table(index=['group', 'asset'], columns='variant', values='sharpe')
    for g in w.index.levels[0]:
        x = w.loc[g]
        for ref in ('ORIGINAL', 'BUY & HOLD'):
            for hv in ('HYBRID', 'HYBRID-REENTRY (exploratory)'):
                k = int((x[hv] > x[ref]).sum()); n = len(x)
                print(g, hv, 'Sharpe >', ref, f'{k}/{n}', 'p=%.2g' % binomtest(k, n).pvalue)


if __name__ == '__main__':
    main()

"""Stage 32: liquidation flush-buy (H1) and crowded-funding exit (H2) as overlays on SuperTrend + vol sizing
(PREREG_stage32_liquidations_positioning.md).  python -m tc.stage32 tune|test"""
import json
import sys
import itertools
import numpy as np
import pandas as pd
from . import stage29 as s29
from . import stage30 as s30
from .stage30b import vol_lev
from .evaluate import supertrend

COINS = s30.EVAL
COST = 0.001
H1_DEV = (pd.Timestamp('2022-12-01', tz='UTC'), pd.Timestamp('2024-01-01', tz='UTC'))
H2_DEV = (pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2024-01-01', tz='UTC'))
T0, T1 = pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2026-10-01', tz='UTC')
H1_GRID = list(itertools.product((2, 3, 4), (3, 7, 14)))
H2_GRID = list(itertools.product((0.80, 0.90, 0.95), (0.0, 0.5)))


def load():
    D = {c: (s30.bars4h(pd.read_parquet(f'data/binance_1h/{c}.parquet')), None) for c in COINS}
    Q = {n: pd.read_parquet(f'data/coinalyze/{n}.parquet') for n in ('liq_long', 'liq_short', 'oi', 'funding')}
    base = {}
    for c in COINS:
        df = D[c][0]
        st = (supertrend(df, 48, 5.0) == 1).astype(float)
        base[c] = dict(st=st, lev=vol_lev(df))
    return D, Q, base


def to_bars(daily, idx):
    """daily value of day d is usable from d+1 00:00 UTC: map each 4h bar (open time) to the previous day."""
    s = daily.copy()
    s.index = s.index.normalize() + pd.Timedelta(days=1)
    s = s[~s.index.duplicated(keep='last')]
    return s.reindex(idx.normalize()).values


def flush_days(Q, c, z):
    x = (Q['liq_long'][c] / Q['oi'][c]).replace([np.inf, -np.inf], np.nan)
    m, sd = x.rolling(90, min_periods=60).mean(), x.rolling(90, min_periods=60).std()
    return ((x - m) / sd > z).astype(float)


def crowded_days(Q, c, q):
    f = Q['funding'][c]
    return (f > f.rolling(365, min_periods=180).quantile(q)).astype(float)


def h1(D, Q, base, z, K):
    P = {}
    for c in COINS:
        fl = flush_days(Q, c, z)
        act = fl.rolling(K, min_periods=1).max()        # a flush on any of the last K known days
        a = np.nan_to_num(to_bars(act, D[c][0].index)) > 0
        b = base[c]
        P[c] = np.where(a, b['lev'], b['st'] * b['lev'])
    return P


def h2(D, Q, base, q, s, unsized=False):
    P = {}
    for c in COINS:
        cr = np.nan_to_num(to_bars(crowded_days(Q, c, q), D[c][0].index)) > 0
        b = base[c]
        a = b['st'] if unsized else b['st'] * b['lev']
        P[c] = np.where(cr, s * a, a)
    return P


def sharpe(P, D, lo, hi):
    return s29.sharpe_d(s29.portfolio(P, D, lo, hi, COST)[1])


def tune():
    D, Q, base = load()
    A = {c: base[c]['st'] * base[c]['lev'] for c in COINS}
    rows = [dict(h='A', z=None, K=None, q=None, s=None, dev='H1', sharpe=sharpe(A, D, *H1_DEV)),
            dict(h='A', z=None, K=None, q=None, s=None, dev='H2', sharpe=sharpe(A, D, *H2_DEV))]
    for z, K in H1_GRID:
        rows.append(dict(h='H1', z=z, K=K, dev='H1', sharpe=sharpe(h1(D, Q, base, z, K), D, *H1_DEV)))
    for q, s in H2_GRID:
        rows.append(dict(h='H2', q=q, s=s, dev='H2', sharpe=sharpe(h2(D, Q, base, q, s), D, *H2_DEV)))
    R = pd.DataFrame(rows)
    R.to_csv('results_stage32_dev.csv', index=False)
    b1 = R[R.h == 'H1'].sort_values('sharpe', ascending=False).iloc[0]
    b2 = R[R.h == 'H2'].sort_values('sharpe', ascending=False).iloc[0]
    fz = dict(H1=dict(z=float(b1.z), K=int(b1.K), dev_sharpe=float(b1.sharpe)),
              H2=dict(q=float(b2.q), s=float(b2.s), dev_sharpe=float(b2.sharpe)),
              A_dev_sharpe_H1=float(R[(R.h == 'A') & (R.dev == 'H1')].sharpe.iloc[0]),
              A_dev_sharpe_H2=float(R[(R.h == 'A') & (R.dev == 'H2')].sharpe.iloc[0]))
    json.dump(fz, open('prereg/FROZEN_stage32.json', 'w'), indent=1)
    print(R.round(3).to_string())
    print(json.dumps(fz, indent=1))


def test():
    D, Q, base = load()
    fz = json.load(open('prereg/FROZEN_stage32.json'))
    A = {c: base[c]['st'] * base[c]['lev'] for c in COINS}
    S = {'A: SuperTrend + vol sizing': A,
         'A + H1 flush-buy': h1(D, Q, base, fz['H1']['z'], fz['H1']['K']),
         'A + H2 crowded exit': h2(D, Q, base, fz['H2']['q'], fz['H2']['s']),
         'SuperTrend': {c: base[c]['st'] for c in COINS},
         'SuperTrend + H2': h2(D, Q, base, fz['H2']['q'], fz['H2']['s'], unsized=True)}
    rows, coin_rows, year_rows, daily = [], [], [], {}
    for name, P in S.items():
        bar, d = s29.portfolio(P, D, T0, T1, COST)
        daily[name] = d
        w = {c: s29.window(D[c][0].index, T0, T1) for c in COINS}
        rows.append(dict(strategy=name, **s29.money(bar, d),
                         turnover_y=float(np.mean([np.abs(np.diff(P[c][w[c]])).sum() / (w[c].sum() / s29.BPY) for c in COINS])),
                         mean_pos=float(np.mean([P[c][w[c]].mean() for c in COINS]))))
        for c in COINS:
            cb, cd = s29.portfolio({c: P[c]}, {c: D[c]}, T0, T1, COST)
            coin_rows.append(dict(strategy=name, coin=c, **s29.money(cb, cd)))
        for Y in (2024, 2025, 2026):
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), min(pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'), T1)
            b2, d2 = s29.portfolio(P, D, lo, hi, COST)
            year_rows.append(dict(strategy=name, year=Y, **s29.money(b2, d2)))
    R = pd.DataFrame(rows); C = pd.DataFrame(coin_rows)
    R.to_csv('results_stage32_test.csv', index=False)
    C.to_csv('results_stage32_per_coin.csv', index=False)
    pd.DataFrame(year_rows).to_csv('results_stage32_years.csv', index=False)
    st = {}
    cw = C.pivot(index='coin', columns='strategy', values='sharpe')
    for h, name in (('H1', 'A + H1 flush-buy'), ('H2', 'A + H2 crowded exit')):
        x, y = daily[name], daily['A: SuperTrend + vol sizing']
        j = x.index.intersection(y.index)
        d0, p, lo5, hi95 = s29.stationary_bootstrap_diff(x[j].values, y[j].values)
        n = int((cw[name] > cw['A: SuperTrend + vol sizing']).sum())
        st[h] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95], coins_better=n,
                     met=bool(d0 > 0 and p < 0.025 and n >= 6))
    x, y = daily['SuperTrend + H2'], daily['SuperTrend']
    j = x.index.intersection(y.index)
    d0, p, lo5, hi95 = s29.stationary_bootstrap_diff(x[j].values, y[j].values)
    st['secondary: SuperTrend + H2 - SuperTrend'] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95])
    # how often the overlays act in the test window
    days = pd.date_range(T0, T1 - pd.Timedelta(days=1), freq='D')
    st['flush days per coin (test, mean)'] = float(np.mean([flush_days(Q, c, fz['H1']['z']).reindex(days).sum() for c in COINS]))
    st['crowded days per coin (test, mean)'] = float(np.mean([crowded_days(Q, c, fz['H2']['q']).reindex(days).sum() for c in COINS]))
    json.dump(st, open('results_stage32_stats.json', 'w'), indent=1)
    pd.set_option('display.width', 200)
    print(R.round(3).to_string())
    print(pd.DataFrame(year_rows).pivot(index='strategy', columns='year', values='sharpe').round(2))
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    {'tune': tune, 'test': test}[sys.argv[1]]()

"""Stage 21: frozen stage-20 model as a long-only trend filter on 10 large caps (PREREG_stage21).
python -m tc.stage21 devprobs|tune|test"""
import itertools
import json
import sys
import numpy as np
import pandas as pd
from . import stage20 as s20
from .stage16 import run_book, alpha, stats, CD, TZ

COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT', 'TRXUSDT']
COST, BPY = 0.001, 365
T0, T1 = pd.Timestamp('2022-01-01', tz=TZ), pd.Timestamp('2026-09-01', tz=TZ)
BENCH = ['EMA50', 'EMA200', 'GOLDEN', 'MACD', 'TSMOM30', 'TSMOM90', 'DONCHIAN20_10', 'SUPERTREND10_3', 'ADX25_DI']
GRID = [(sp, ti, to) for sp in (1, 3, 7, 14) for ti in (0.5, 0.55, 0.6) for to in (0.4, 0.45, 0.5) if to <= ti]


def devprobs():
    p = json.load(open('FROZEN_stage20.json'))['params']
    X, D = s20.load()
    parts = []
    for Y in (2019, 2020, 2021):
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
        m, cols = s20.fit(X, p['H'], lo, p)
        parts.append(s20.predict(m, cols, X, lo, hi))
        print(Y, flush=True)
    pd.concat(parts).to_pickle(f'{CD}/stage21_devprobs.pkl')


def state(P, span, tin, tout):
    s = P.ewm(span=span, adjust=False, ignore_na=True).mean() if span > 1 else P
    v = s.values
    out = np.zeros(v.shape, bool)
    cur = np.zeros(v.shape[1], bool)
    for i in range(len(v)):
        x = v[i]
        cur = np.where(np.isnan(x), cur, np.where(cur, x >= tout, x > tin))
        out[i] = cur
    return pd.DataFrame(out, index=P.index, columns=P.columns)


def ctx(D, lo, hi):
    rex = D['rex'].loc[lo:hi, COINS]
    live = rex.notna() & D['C'].loc[lo:hi, COINS].notna()
    return rex, live


def portfolio(mask, rex, live):
    w = (mask.reindex(index=rex.index, columns=COINS).fillna(False) & live).astype(float) * 0.1
    return run_book(w, rex, COST, 1)


def bench(D, rex, live):
    R = {k: portfolio(D['ind'][k][COINS].loc[rex.index], rex, live)[0] for k in BENCH}
    R['BUY_HOLD'] = portfolio(pd.DataFrame(True, index=rex.index, columns=COINS), rex, live)[0]
    return pd.DataFrame(R)


def tune():
    X, D = s20.load()
    P = pd.read_pickle(f'{CD}/stage21_devprobs.pkl').unstack()[COINS]
    rows = []
    for cfg in GRID:
        ts, shs = [], []
        for Y in (2019, 2020, 2021):
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
            rex, live = ctx(D, lo, hi)
            st = state(P.loc[lo:hi].reindex(rex.index), *cfg)
            r, _ = portfolio(st, rex, live)
            a, t, _ = alpha(r, bench(D, rex, live))
            ts.append(t if np.isfinite(t) else 0.0)
            shs.append(stats(r, BPY)['sharpe'])
        rows.append(dict(span=cfg[0], th_in=cfg[1], th_out=cfg[2], t_mean=float(np.mean(ts)), t_folds=ts, sharpe_folds=shs))
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values('t_mean', ascending=False)
    b = T.iloc[0]
    json.dump(dict(params=dict(span=int(b.span), th_in=float(b.th_in), th_out=float(b.th_out)), dev_t=float(b.t_mean),
                   grid=T.to_dict('records'), frozen_at=str(pd.Timestamp.utcnow())),
              open('FROZEN_stage21.json', 'w'), indent=1, default=str)
    print('BEST', b.to_dict())


def quality(mask, D, rex, live):
    C = D['C'].astype(float)[COINS]
    out = {}
    lc = np.log(C)
    f7 = (lc.shift(-7) - lc).loc[rex.index]
    f30 = (lc.shift(-30) - lc).loc[rex.index]
    m = mask.reindex(index=rex.index, columns=COINS).fillna(False)
    L = live & f30.notna()
    on, off = m & L, ~m & L
    out['fwd7_on'] = float(np.expm1(f7[on].stack()).mean())
    out['fwd7_off'] = float(np.expm1(f7[off].stack()).mean())
    out['fwd30_on'] = float(np.expm1(f30[on].stack()).mean())
    out['fwd30_off'] = float(np.expm1(f30[off].stack()).mean())
    crash = (f30 < np.log(0.8)) & L
    rally = (f30 > np.log(1.2)) & L
    out['crashes_avoided'] = float((off & crash).values.sum() / max(crash.values.sum(), 1))
    out['rallies_captured'] = float((on & rally).values.sum() / max(rally.values.sum(), 1))
    flips = (m.astype(int).diff().abs() > 0) & live
    yrs = live.sum() / 365
    out['flips_per_coin_year'] = float((flips.sum() / yrs.replace(0, np.nan)).mean())
    out['time_in_market'] = float(on.values.sum() / max(L.values.sum(), 1))
    return out


def test():
    p = json.load(open('FROZEN_stage21.json'))['params']
    X, D = s20.load()
    rex, live = ctx(D, T0, T1)
    P = pd.read_pickle(f'{CD}/crypto20_probs.pkl').unstack()[COINS].reindex(rex.index)
    st = state(P, p['span'], p['th_in'], p['th_out'])
    r, to = portfolio(st, rex, live)
    B = bench(D, rex, live)
    a, t, m = alpha(r, B)
    rows = [dict(filter='MODEL', alpha_ann=a * BPY, alpha_t=t, **stats(r, BPY), **quality(st, D, rex, live))]
    masks = {k: D['ind'][k][COINS].loc[rex.index] for k in BENCH}
    masks['BUY_HOLD'] = pd.DataFrame(True, index=rex.index, columns=COINS)
    for k in B:
        rows.append(dict(filter=k, **stats(B[k], BPY), **quality(masks[k], D, rex, live)))
    T = pd.DataFrame(rows)
    # per-coin Sharpe of the filtered coin
    pc = []
    for c in COINS:
        rc = rex[c].where(live[c])
        row = dict(coin=c)
        for k, mk in [('MODEL', st)] + list(masks.items()):
            x = (mk[c].astype(float).reindex(rex.index).fillna(0) * rc).dropna()
            row[k] = stats(x - COST * mk[c].astype(float).reindex(x.index).diff().abs().fillna(0), BPY)['sharpe']
        pc.append(row)
    PC = pd.DataFrame(pc)
    wins = {k: int((PC.MODEL > PC[k]).sum()) for k in list(masks)}
    Cc = pd.concat([r.rename('MODEL'), B], axis=1)
    yr = Cc.groupby(Cc.index.year).apply(lambda g: np.expm1(np.log1p(g).sum()))
    T.to_csv('results_stage21_filter.csv', index=False)
    PC.to_csv('results_stage21_filter_per_coin.csv', index=False)
    yr.to_csv('results_stage21_filter_years.csv')
    Cc.to_pickle(f'{CD}/stage21_returns.pkl')
    st.to_pickle(f'{CD}/stage21_states.pkl')
    pd.set_option('display.width', 260)
    print(T.round(3).to_string())
    print(PC.round(2).to_string())
    print('per-coin Sharpe wins of MODEL vs each:', wins)
    print(yr.round(2).to_string())
    ok1 = a > 0 and t >= 2.5
    ok2 = T.sharpe.iloc[0] > T.sharpe.iloc[1:].max()
    print('P1 alpha t>=2.5:', ok1, ' P2 Sharpe > every filter:', ok2, ' PRIMARY:', bool(ok1 and ok2))


if __name__ == '__main__':
    dict(devprobs=devprobs, tune=tune, test=test)[sys.argv[1]]()

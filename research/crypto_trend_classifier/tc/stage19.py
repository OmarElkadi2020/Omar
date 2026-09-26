"""Stage 19: long-only rotation among 5 user-chosen coins with frozen stage-16 scores (PREREG_stage19)."""
import json
import numpy as np
import pandas as pd
from .stage16 import load, run_book, alpha, stats, CD

COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'BNBUSDT']
T0, T1 = pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2026-09-01', tz='UTC')


def topk(S, k):
    rk = S.rank(axis=1, ascending=False)
    w = (rk <= k).astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def main():
    p = json.load(open('FROZEN_stage16_crypto.json'))['params']
    X, D = load('crypto')
    rex = D['rex'].loc[T0:T1, COINS]
    Sall = pd.read_pickle(f'{CD}/crypto_scores.pkl').unstack().reindex(index=rex.index)
    pct = Sall.rank(axis=1, pct=True)[COINS]           # rank inside the full top-100 universe
    S = Sall[COINS]
    ok = S.notna().sum(axis=1) == 5
    S, pct, rex = S[ok], pct[ok], rex[ok]
    ew = rex.mean(axis=1)
    btc = rex['BTCUSDT']
    F = pd.DataFrame({'EW5': ew, 'BTC': btc})
    rows, curves = [], {}

    def add(name, w):
        r, to = run_book(w, rex, 0.001, p['hold'])
        a, t, _ = alpha(r, F)
        rows.append(dict(variant=name, alpha_ann=a * 365, alpha_t=t, turnover_day=float(to.mean()), **stats(r, 365)))
        curves[name] = r

    add('PRIMARY top-2 of 5', topk(S, 2))
    add('top-1 of 5', topk(S, 1))
    add('top-3 of 5', topk(S, 3))
    w = topk(S, 2) * (pct > 0.8)                       # slot goes to cash if the coin is not top-20% of the universe
    add('top-2 + cash filter (universe top 20%)', w)
    for name, x in (('EW 5 coins (rebalanced daily)', ew), ('BTC buy & hold', btc)):
        rows.append(dict(variant=name, **stats(x, 365)))
        curves[name] = x
    R = pd.DataFrame(rows)
    C = pd.DataFrame(curves)
    yr = C.groupby(C.index.year).apply(lambda g: np.expm1(np.log1p(g.clip(lower=-0.99)).sum()))
    R.to_csv('results_stage19_5coins.csv', index=False)
    yr.to_csv('results_stage19_5coins_years.csv')
    C.to_pickle(f'{CD}/stage19_curves.pkl')
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())
    print(yr.round(2).to_string())
    print('held share per coin (primary):', (topk(S, 2) > 0).mean().round(2).to_dict())
    print('PRIMARY:', bool(R.iloc[0].alpha_ann > 0 and R.iloc[0].alpha_t >= 2))


if __name__ == '__main__':
    main()

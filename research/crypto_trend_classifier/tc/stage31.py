"""Stage 31: forward test of frozen rules on data after 2026-10-01 (PREREG_stage31_forward_test.md).
python -m tc.stage31 update   # append new Binance 1h data (monthly, then daily files)
python -m tc.stage31 run      # positions with the frozen procedure, money report on the forward window"""
import json
import sys
import numpy as np
import pandas as pd
from . import stage29 as s29
from . import stage30 as s30
from .stage30b import vol_lev, MODELS
from .evaluate import supertrend

F0 = pd.Timestamp('2026-10-01', tz='UTC')


def load_policy(year):
    z = np.load(f'{MODELS}/policy_lf_{year}.npz')
    models = []
    for i in range(len(s30.SEEDS)):
        m = s30.MLP(1, 1, 0)
        m.set([z[f'W1_{i}'], z[f'b1_{i}'], z[f'w2_{i}'], np.atleast_1d(z[f'b2_{i}'])])
        models.append(m)
    return models, z['mu'], z['sd']


def policy_positions(D, end):
    """frozen procedure: the model of year Y is trained on data before 1 Jan Y (2026 model is the saved one)."""
    ch = json.load(open('prereg/FROZEN_stage30.json'))['choice']['lf']
    P = {c: np.zeros(len(D[c][0])) for c in s30.EVAL}
    for Y in range(2026, end.year + 1):
        cut = pd.Timestamp(f'{Y}-01-01', tz='UTC')
        if Y == 2026:
            models, mu, sd = load_policy(2026)
        else:
            models, mu, sd, _ = s30.fit(cut, 'lf', ch['hidden'], ch['l2'], ch['c_train'])
            from .stage30b import save_models
            save_models(models, mu, sd, 'lf', cut)
        for c in s30.EVAL:
            idx = D[c][0].index
            w = (idx >= cut) & (idx < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))
            P[c][w] = s30.predict(models, mu, sd, D[c][1].values[w], 'lf')
    return P


def run():
    s30._D = None
    D = s30.data()
    end = max(D[c][0].index[-1] for c in s30.EVAL) + s29.STEP
    POL = policy_positions(D, end)
    LEV = {c: vol_lev(D[c][0]) for c in s30.EVAL}
    ST = {c: (supertrend(D[c][0], *s29.ST) == 1).astype(float) for c in s30.EVAL}
    S = {'A: SuperTrend + vol sizing': {c: ST[c] * LEV[c] for c in s30.EVAL},
         'B: POLICY': POL,
         'C: POLICY direction, vol sized': {c: (POL[c] >= 0.5) * LEV[c] for c in s30.EVAL},
         'SuperTrend': ST, 'buy & hold': {c: np.ones(len(D[c][0])) for c in s30.EVAL}}
    rows, daily = [], {}
    for name, P in S.items():
        bar, d = s29.portfolio(P, D, F0, end, s30.COST)
        daily[name] = d
        rows.append(dict(strategy=name, days=len(d), **s29.money(bar, d)))
    R = pd.DataFrame(rows)
    st = {}
    for a, b in (('C: POLICY direction, vol sized', 'A: SuperTrend + vol sizing'), ('B: POLICY', 'A: SuperTrend + vol sizing')):
        x, y = daily[a], daily[b]
        j = x.index.intersection(y.index)
        if len(j) > 40:
            d0, p, lo5, hi95 = s29.stationary_bootstrap_diff(x[j].values, y[j].values)
            st[f'{a} - {b}'] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95], days=len(j))
    R.to_csv('results_stage31_forward.csv', index=False)
    json.dump(dict(end=str(end), **st), open('results_stage31_forward.json', 'w'), indent=1)
    print('forward window', F0.date(), '…', end)
    print(R.round(3).to_string())
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'update':
        from .download_binance import COINS, update
        for c in COINS:
            print(*update(c), flush=True)
    else:
        run()

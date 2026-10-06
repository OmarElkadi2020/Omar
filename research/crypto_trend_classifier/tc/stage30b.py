"""Stage 30b (EXPLORATORY, the stage-30 test window was already seen): where does POLICY's edge come from?
Compares POLICY with volatility-sized rules (stage-27 recipe) and decomposes its position.
Also saves the frozen 2026 policy models (trained on data before 2026-01-01) for the forward test.
python -m tc.stage30b diag"""
import json
import os
import numpy as np
import pandas as pd
from . import stage29 as s29
from . import stage30 as s30
from .evaluate import supertrend

MODELS = 'models/stage30'


def vol_lev(df):
    """stage-27 sizing: 2 % daily vol target from a trailing 30-day std of 4h returns, capped at 2x."""
    c = df.close.values
    r = np.r_[np.diff(c) / c[:-1], 0.0]
    n = 180
    sig = pd.Series(np.r_[0.0, r[:-1]]).rolling(n, min_periods=n // 3).std().values
    return np.nan_to_num(np.minimum(0.02 * np.sqrt(4 / 24) / sig, 2.0))


def save_models(models, mu, sd, mode, cut):
    os.makedirs(MODELS, exist_ok=True)
    np.savez_compressed(f'{MODELS}/policy_{mode}_{cut:%Y}.npz', mu=mu, sd=sd,
                        **{f'{k}_{i}': v for i, m in enumerate(models) for k, v in
                           zip(('W1', 'b1', 'w2', 'b2'), m.params())})


def diag():
    D = s30.data()
    fz = json.load(open('prereg/FROZEN_stage30.json'))['choice']
    ch = fz['lf']
    POL = {c: np.zeros(len(D[c][0])) for c in s30.EVAL}
    for Y in s29.TEST_YEARS:
        cut = pd.Timestamp(f'{Y}-01-01', tz='UTC')
        models, mu, sd, _ = s30.fit(cut, 'lf', ch['hidden'], ch['l2'], ch['c_train'])
        if Y == s29.TEST_YEARS[-1]:
            save_models(models, mu, sd, 'lf', cut)
        for c in s30.EVAL:
            idx = D[c][0].index
            w = (idx >= cut) & (idx < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))
            POL[c][w] = s30.predict(models, mu, sd, D[c][1].values[w], 'lf')
        print('fitted', Y, flush=True)
    LEV = {c: vol_lev(D[c][0]) for c in s30.EVAL}
    ST = {c: (supertrend(D[c][0], *s29.ST) == 1).astype(float) for c in s30.EVAL}
    S = {'POLICY': POL,
         'SuperTrend + vol sizing': {c: ST[c] * LEV[c] for c in s30.EVAL},
         'buy & hold + vol sizing': LEV,
         'SuperTrend': ST,
         'buy & hold': {c: np.ones(len(D[c][0])) for c in s30.EVAL},
         'POLICY direction, vol sized': {c: (POL[c] >= 0.5) * LEV[c] for c in s30.EVAL},
         'POLICY rescaled to vol target': {c: POL[c] / max(np.mean(POL[c][POL[c] > 0]), 1e-9) * LEV[c] for c in s30.EVAL}}
    rows, daily = [], {}
    for name, P in S.items():
        bar, d = s29.portfolio(P, D, s30.T0, s30.T1, s30.COST)
        daily[name] = d
        w = {c: s29.window(D[c][0].index, s30.T0, s30.T1) for c in s30.EVAL}
        rows.append(dict(strategy=name, **s29.money(bar, d),
                         mean_pos=float(np.mean([P[c][w[c]].mean() for c in s30.EVAL])),
                         turnover_y=float(np.mean([np.abs(np.diff(P[c][w[c]])).sum() / (w[c].sum() / s29.BPY) for c in s30.EVAL]))))
    R = pd.DataFrame(rows)
    st = {}
    for b in ('SuperTrend + vol sizing', 'buy & hold + vol sizing', 'SuperTrend'):
        x, y = daily['POLICY'], daily[b]
        j = x.index.intersection(y.index)
        d0, p, lo5, hi95 = s29.stationary_bootstrap_diff(x[j].values, y[j].values)
        st[f'POLICY - {b}'] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95])
    # what explains the position: vol-sizing leverage and SuperTrend state (test bars, pooled coins)
    X, y = [], []
    for c in s30.EVAL:
        w = s29.window(D[c][0].index, s30.T0, s30.T1)
        X.append(np.c_[np.ones(w.sum()), LEV[c][w], ST[c][w]]); y.append(POL[c][w])
    X, y = np.vstack(X), np.concatenate(y)
    for name, cols in (('vol leverage only', [0, 1]), ('SuperTrend only', [0, 2]), ('both', [0, 1, 2])):
        b = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
        st[f'R2 position ~ {name}'] = float(1 - np.var(y - X[:, cols] @ b) / np.var(y))
    st['corr(position, vol leverage)'] = float(np.corrcoef(y, X[:, 1])[0, 1])
    st['corr(position, SuperTrend up)'] = float(np.corrcoef(y, X[:, 2])[0, 1])
    R.to_csv('results_stage30b_diag.csv', index=False)
    json.dump(st, open('results_stage30b_diag.json', 'w'), indent=1)
    pd.set_option('display.width', 200)
    print(R.round(3))
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    diag()

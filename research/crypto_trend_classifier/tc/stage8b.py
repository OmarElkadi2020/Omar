"""Stage 8b: FX carry / cross-sectional filters (see PREREG_stage8b_fx_carry.md). Run once."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .stage8 import prices, candidates, sharpe, stats, boot_diff, sl, COST, BPY

DEV, TEST = ('1974-01-01', '2004-12-31'), ('2005-01-01', '2020-12-31')
ISO = {'AUD': 'AUS', 'CAD': 'CAN', 'JPY': 'JPN', 'NOK': 'NOR', 'SEK': 'SWE', 'CHF': 'CHE', 'GBP': 'GBR'}
U = ['AUD', 'CAD', 'EUR', 'JPY', 'NOK', 'SEK', 'CHF', 'GBP']


def rates(idx):
    d = pd.read_excel('fxc/data/sources/jst/jst_macrohistory.xlsx')
    s = d.pivot(index='year', columns='iso', values='stir')
    out = pd.DataFrame(index=idx)
    y = idx.year - 1                                   # previous calendar year only (causal)
    for c in U:
        if c == 'EUR':
            v = np.where(idx < '1999-01-04', s['DNK'].reindex(y).values, s['DEU'].reindex(y).values)
        else:
            v = s[ISO[c]].reindex(y).values
        out[c] = v
    out['USD'] = s['USA'].reindex(y).values
    return out


def xs_rank_sig(score, k=3):
    r = score.rank(axis=1, ascending=False)
    n = score.notna().sum(axis=1)
    sig = pd.DataFrame(0.0, score.index, score.columns)
    sig[r <= k] = 1.0
    sig[r.gt(n - k, axis=0) & score.notna()] = -1.0
    return sig.where(n >= 2 * k, 0.0, axis=0)


def portfolio(sig, P, carry, cost=COST, delay=0):
    r = (np.log(P).diff() + carry).shift(-1)
    vol = np.log(P).diff().rolling(60).std() * np.sqrt(BPY)
    w = (sig / vol * 0.10).shift(delay)
    w = w.where(np.isfinite(w), 0.0).fillna(0.0)
    n = len(P.columns)
    pnl = (w * r.fillna(0)).sum(axis=1) / n - cost * w.diff().abs().sum(axis=1) / n
    return pnl


def main():
    P = prices()[U]
    P = P[P.index >= '1972-01-01']
    R = rates(P.index)
    diff = R[U].sub(R['USD'], axis=0)
    carry = diff / 100 / BPY
    C8 = candidates(P)
    lp = np.log(P)
    trend = {'TSMOM 63d': np.sign(lp - lp.shift(63)), 'TSMOM 252d': np.sign(lp - lp.shift(252)),
             'EMA 32/128': C8['EMA 32/128'], 'Baz': np.sign(C8['Continuous trend (Baz)'])}
    cs = xs_rank_sig(diff.where(P.notna()))
    C = {'Carry 3x3': cs}
    for k, t in trend.items():
        C[f'Carry filtered by {k}'] = cs.where(cs * t > 0, 0.0)
    for L in (21, 63, 126, 252):
        C[f'XS momentum {L}d'] = xs_rank_sig(lp - lp.shift(L))
    C['Carry + XS mom 63d'] = 0.5 * cs + 0.5 * C['XS momentum 63d']
    avg = diff.mean(axis=1)
    C['Dollar carry'] = pd.DataFrame(np.sign(avg).values[:, None] * np.ones((1, len(U))), P.index, U)
    bh = portfolio(pd.DataFrame(1.0, P.index, U), P, carry, cost=0.0)
    pn, rows = {}, []
    for k, s in C.items():
        pn[k] = portfolio(s, P, carry)
        rows.append(dict(filter=k, dev_sharpe=sharpe(sl(pn[k], DEV)), test_sharpe=sharpe(sl(pn[k], TEST)),
                         test_2005_12=sharpe(sl(pn[k], ('2005-01-01', '2012-12-31'))),
                         test_2013_20=sharpe(sl(pn[k], ('2013-01-01', '2020-12-31')))))
    T = pd.DataFrame(rows).sort_values('dev_sharpe', ascending=False)
    best = T.iloc[0]['filter']
    te, bte = sl(pn[best], TEST), sl(bh, TEST)
    ci_diff, ci_own = boot_diff(te, bte)
    N = 25
    sr_sd = pd.concat([T.dev_sharpe, pd.Series([1.111, 1.056, 1.039, 1.006, .932, .895, .888, .851, .838, .812, .808, .791, .694, .687])]).std() / np.sqrt(BPY)
    emax = sr_sd * ((1 - 0.5772) * norm.ppf(1 - 1 / N) + 0.5772 * norm.ppf(1 - 1 / (N * np.e))) * np.sqrt(BPY)
    res = dict(selected=best, dev=stats(sl(pn[best], DEV)), test=stats(te), buy_hold_test=stats(bte),
               buy_hold_dev=stats(sl(bh, DEV)), ci90_sharpe_diff_vs_bh=ci_diff, ci90_own_sharpe=ci_own,
               expected_max_noise_sharpe_25_trials=emax,
               cost_x3=sharpe(sl(portfolio(C[best], P, carry, cost=3 * COST), TEST)),
               delay_1d=sharpe(sl(portfolio(C[best], P, carry, delay=1), TEST)),
               carry_share_of_bh_test=float(sl((carry.mean(axis=1)), TEST).sum()))
    pd.to_pickle(dict(res=res, table=T, pnl=pn, bh=bh), 'out_stage8b.pkl')
    pd.set_option('display.width', 200)
    print(T.round(3).to_string(index=False))
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

"""Stage 8: best FX trend filter, G10 vs USD daily 1971-2026 (see PREREG_stage8_fx.md). Run once."""
import json
import numpy as np
import pandas as pd

CCY = {'Australia': 'AUD', 'Canada': 'CAD', 'Euro': 'EUR', 'Japan': 'JPY', 'New Zealand': 'NZD', 'Norway': 'NOK',
       'Sweden': 'SEK', 'Switzerland': 'CHF', 'United Kingdom': 'GBP'}
DEV = ('1974-01-01', '2004-12-31')
TEST = ('2005-01-01', '2026-12-31')
COST, BPY = 0.0003, 252


def prices():
    d = pd.read_csv('data/fx_daily_fred.csv', parse_dates=['Date'])
    P = d.pivot_table(index='Date', columns='Country', values='Exchange rate')
    P = 1.0 / P.where(P > 0)                        # USD per foreign unit
    eur = P['Euro'].copy()
    dkk = P['Denmark']
    # splice: DKK returns before 1999-01-04, EUR after
    r = np.log(dkk).diff().where(eur.isna(), np.log(eur).diff())
    P['Euro'] = np.exp(r.fillna(0).cumsum())
    P = P[list(CCY)].rename(columns=CCY).ffill(limit=5)
    return P


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def phi(x):
    return x * np.exp(-x ** 2 / 4) / 0.89


def candidates(P):
    lp = np.log(P)
    C = {}
    for L in (21, 63, 126, 252):
        C[f'TSMOM {L}d'] = np.sign(lp - lp.shift(L))
    for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)):
        C[f'EMA {f}/{s}'] = np.sign(ema(P, f) - ema(P, s))
    for N in (20, 55, 100, 250):
        hi, lo = P.rolling(N).max(), P.rolling(N).min()
        sig = pd.DataFrame(np.where(P >= hi, 1.0, np.where(P <= lo, -1.0, np.nan)), P.index, P.columns)
        C[f'Donchian {N}'] = sig.ffill().fillna(0) * hi.notna()
    parts = []
    for f in (8, 16, 32):
        s = 3 * f
        x = (ema(P, f) - ema(P, s)) / P.rolling(63).std()
        x = x / x.rolling(252).std()
        parts.append(phi(x))
    C['Continuous trend (Baz)'] = (sum(parts) / 3).clip(-1, 1)
    rules = [k for k in C]
    C['Ensemble (all 12 rules)'] = sum(C[k] for k in rules if k != 'Continuous trend (Baz)') / 12
    return C


def portfolio(sig, P, cost=COST, delay=0):
    r = np.log(P).diff().shift(-1)                   # return from close t to close t+1
    vol = np.log(P).diff().rolling(60).std() * np.sqrt(BPY)
    w = (sig / vol * 0.10).shift(delay)              # 10% annual vol target per currency
    w = w.where(np.isfinite(w), 0.0).fillna(0.0)
    n = r.notna().sum(axis=1).clip(lower=1)
    pnl = (w * r.fillna(0)).sum(axis=1) / n - cost * w.diff().abs().sum(axis=1) / n
    return pnl, w


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(BPY) if x.std() > 0 else np.nan


def stats(pnl):
    eq = pnl.cumsum()
    dd = (np.exp(eq - eq.cummax()) - 1).min()
    yrs = len(pnl) / BPY
    return dict(sharpe=sharpe(pnl), ann_ret=np.exp(pnl.sum() / yrs) - 1, ann_vol=pnl.std() * np.sqrt(BPY), max_dd=dd)


def boot_diff(a, b, block=63, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = a.values, b.values
    n, k = len(a), int(np.ceil(len(a) / block))
    out, own = [], []
    for _ in range(reps):
        ix = (rng.integers(0, n - block, k)[:, None] + np.arange(block)).ravel()[:n]
        sa, sb = a[ix].mean() / a[ix].std(), b[ix].mean() / b[ix].std()
        out.append((sa - sb) * np.sqrt(BPY))
        own.append(sa * np.sqrt(BPY))
    return np.quantile(out, [0.05, 0.95]).tolist(), np.quantile(own, [0.05, 0.95]).tolist()


def sl(x, per):
    return x[(x.index >= per[0]) & (x.index <= per[1])]


def main():
    P = prices()
    C = candidates(P)
    bh_sig = pd.DataFrame(1.0, P.index, P.columns).where(P.notna())
    bh, _ = portfolio(bh_sig, P, cost=0.0)
    rows = []
    pn = {}
    for k, s in C.items():
        pnl, _ = portfolio(s, P)
        pn[k] = pnl
        rows.append(dict(filter=k, dev_sharpe=sharpe(sl(pnl, DEV)), test_sharpe=sharpe(sl(pnl, TEST))))
    R = pd.DataFrame(rows).sort_values('dev_sharpe', ascending=False)
    best = R.iloc[0]['filter']
    s = C[best]
    te = sl(pn[best], TEST)
    bte = sl(bh, TEST)
    ci_diff, ci_own = boot_diff(te, bte)
    # deflated-Sharpe style haircut: expected max Sharpe of 14 noise strategies over the dev sample
    from scipy.stats import norm
    N, T = len(C), len(sl(pn[best], DEV))
    sr_sd = R.dev_sharpe.std() / np.sqrt(BPY)
    emax = sr_sd * ((1 - 0.5772) * norm.ppf(1 - 1 / N) + 0.5772 * norm.ppf(1 - 1 / (N * np.e)))
    res = dict(selected=best, dev=stats(sl(pn[best], DEV)), test=stats(te), buy_hold_test=stats(bte),
               buy_hold_dev=stats(sl(bh, DEV)), ci90_sharpe_diff_vs_bh=ci_diff, ci90_own_sharpe=ci_own,
               expected_max_noise_sharpe_dev=emax * np.sqrt(BPY))
    rob = {}
    rob['cost x3'] = sharpe(sl(portfolio(s, P, cost=3 * COST)[0], TEST))
    rob['delay 1 day'] = sharpe(sl(portfolio(s, P, delay=1)[0], TEST))
    for a, b in (('2005-01-01', '2012-12-31'), ('2013-01-01', '2019-12-31'), ('2020-01-01', '2026-12-31')):
        rob[f'{a[:4]}-{b[:4]}'] = dict(filter=sharpe(sl(pn[best], (a, b))), buy_hold=sharpe(sl(bh, (a, b))))
    res['robustness'] = rob
    per = {}
    for c in P.columns:
        pc, _ = portfolio(s[[c]], P[[c]])
        pb, _ = portfolio(bh_sig[[c]], P[[c]], cost=0.0)
        per[c] = dict(filter=sharpe(sl(pc, TEST)), buy_hold=sharpe(sl(pb, TEST)))
    res['per_currency_test'] = per
    pd.to_pickle(dict(res=res, table=R, pnl=pn, bh=bh), 'out_stage8.pkl')
    pd.set_option('display.width', 200)
    print(R.round(3).to_string(index=False))
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

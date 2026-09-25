"""Stage 9: technical-only stock trend filters (see PREREG_stage9_stocks.md). Run once."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .stage8 import sharpe, boot_diff, sl

DEV, TEST = ('1996-01-01', '2010-12-31'), ('2011-01-01', '2026-12-31')
COST, BPY = 0.001, 252


def load():
    from .stage3 import STOCKS
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'adj_close'])
    d = d[d.ticker.isin(STOCKS)]
    P = d.pivot_table(index='date', columns='ticker', values='adj_close').sort_index()
    P.index = pd.to_datetime(P.index)
    P = P.where(P > 0)
    R = np.log(P).diff().clip(-0.4, 0.4)
    age = P.notna().cumsum()
    elig = (age >= 273) & (P > 1) & R.notna()
    return P, R, elig


def month_end(df):
    me = df.index.to_series().groupby(df.index.to_period('M')).transform('max') == df.index
    out = df.copy()
    out.loc[~me.values] = np.nan
    return out.ffill()


def ew(mask):
    m = mask.astype(float)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def top(score, elig, q, low=False):
    s = score.where(elig)
    r = s.rank(axis=1, pct=True, ascending=not low)
    return r >= 1 - q


def run(W, R, cost=COST, delay=0):
    W = W.shift(delay).fillna(0.0)
    fwd = np.expm1(R.shift(-1)).fillna(0.0)
    gross = (W * fwd).sum(axis=1)
    pnl = np.log1p(gross) - cost * W.diff().abs().sum(axis=1)
    return pnl


def stats(pnl):
    eq = pnl.cumsum()
    dd = np.exp(eq - eq.cummax()) - 1
    yrs = len(pnl) / BPY
    cagr = np.exp(pnl.sum() / yrs) - 1
    return dict(sharpe=sharpe(pnl), cagr=cagr, vol=pnl.std() * np.sqrt(BPY), max_dd=dd.min(), calmar=cagr / abs(dd.min()))


def main():
    P, R, elig = load()
    idx = np.exp(R.where(elig).mean(axis=1).fillna(0).cumsum())
    sma200 = P.rolling(200, min_periods=200).mean()
    sma210 = P.rolling(210, min_periods=210).mean()
    e50, e200 = P.ewm(span=50, adjust=False).mean(), P.ewm(span=200, adjust=False).mean()
    mom = np.log(P.shift(21) / P.shift(252))
    hi52 = P / P.rolling(252, min_periods=252).max()
    vol = R.rolling(252, min_periods=200).std()
    mkt_on = idx > idx.rolling(200).mean()
    br_on = ((P > sma200) & elig).sum(axis=1) / (sma200.notna() & elig).sum(axis=1).replace(0, np.nan) > 0.5
    me = lambda x: month_end(x.astype(float)).fillna(0).astype(bool)  # noqa: E731
    mom_me, hi_me, vol_me = month_end(mom), month_end(hi52), month_end(vol)
    C = {}
    C['1 price>SMA200'] = ew(elig & (P > sma200))
    C['2 price>SMA210 monthly'] = ew(elig & me(P > sma210))
    C['3 EMA50>EMA200'] = ew(elig & (e50 > e200))
    C['4 12-1 return>0 monthly'] = ew(elig & me(mom > 0))
    C['5 market timing: index>SMA200'] = ew(elig).mul(mkt_on.astype(float), axis=0)
    C['6 market timing: breadth>50%'] = ew(elig).mul(br_on.astype(float), axis=0)
    C['7 momentum top decile'] = ew(elig & top(mom_me, elig, 0.1))
    C['8 momentum top decile + market timing'] = C['7 momentum top decile'].mul(mkt_on.astype(float), axis=0)
    C['9 52w-high top decile'] = ew(elig & top(hi_me, elig, 0.1))
    C['10 low-volatility decile'] = ew(elig & top(vol_me, elig, 0.1, low=True))
    C['11 momentum top quintile & price>SMA200'] = ew(elig & top(mom_me, elig, 0.2) & (P > sma200))
    C['12 blend 7+9+10'] = (C['7 momentum top decile'] + C['9 52w-high top decile'] + C['10 low-volatility decile']) / 3
    bench = run(ew(elig), R, cost=0.0)
    pn, rows = {}, []
    for k, W in C.items():
        pn[k] = run(W, R)
        rows.append(dict(filter=k, dev_sharpe=sharpe(sl(pn[k], DEV)), test_sharpe=sharpe(sl(pn[k], TEST)),
                         test_cagr=stats(sl(pn[k], TEST))['cagr'], test_maxdd=stats(sl(pn[k], TEST))['max_dd'],
                         t_2011_18=sharpe(sl(pn[k], ('2011-01-01', '2018-12-31'))),
                         t_2019_26=sharpe(sl(pn[k], ('2019-01-01', '2026-12-31'))),
                         invested=float(sl((W.sum(axis=1) > 0).astype(float), TEST).mean())))
        print(k, flush=True)
    T = pd.DataFrame(rows).sort_values('dev_sharpe', ascending=False)
    best = T.iloc[0]['filter']
    te, bte = sl(pn[best], TEST), sl(bench, TEST)
    ci_diff, ci_own = boot_diff(te, bte)
    N = len(C)
    emax = T.dev_sharpe.std() * ((1 - 0.5772) * norm.ppf(1 - 1 / N) + 0.5772 * norm.ppf(1 - 1 / (N * np.e)))
    # benchmark drawdowns >= 20% in test: how much of each did the filter avoid?
    eq = bte.cumsum()
    dd = np.exp(eq - eq.cummax()) - 1
    eps, i = [], 0
    ddv = dd.values
    while i < len(ddv):
        if ddv[i] <= -0.20:
            a = i
            while a > 0 and ddv[a] < 0:
                a -= 1
            b = i
            while b < len(ddv) and ddv[b] < 0:
                b += 1
            t = int(np.argmin(ddv[a:b])) + a
            eps.append(dict(start=str(dd.index[a].date()), trough=str(dd.index[t].date()), bench_dd=float(ddv[t]),
                            filter_ret=float(np.expm1(te.iloc[a + 1:t + 1].sum()))))
            i = b
        i += 1
    res = dict(selected=best, dev=stats(sl(pn[best], DEV)), test=stats(te), bench_dev=stats(sl(bench, DEV)),
               bench_test=stats(bte), ci90_sharpe_diff=ci_diff, ci90_own=ci_own, expected_max_noise_sharpe=emax,
               cost_x3=sharpe(sl(run(C[best], R, cost=3 * COST), TEST)),
               delay_1d=sharpe(sl(run(C[best], R, delay=1), TEST)), bench_drawdowns_test=eps)
    pd.to_pickle(dict(res=res, table=T, pnl=pn, bench=bench), 'out_stage9.pkl')
    pd.set_option('display.width', 220)
    print(T.round(3).to_string(index=False))
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

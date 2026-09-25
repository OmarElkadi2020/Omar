"""Measure the ORIGINAL Jesse strategy (regime and actual position, 4h approximation) with the stage-13 lag metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from .evaluate import supertrend
from .features import _atr
from .s12lib import lag_metrics
from .stage13 import fr, COINS, HOLDOUT, TEST_END


def adx(f, n=14):
    h, l, c = f.high, f.low, f.close
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, f.index).ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * pd.Series(ndm, f.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=1 / n, adjust=False).mean().values


def strategy_states(f):
    c = f.close.values
    st = np.asarray(supertrend(f, 24, 4.0))
    e10 = f.close.ewm(span=10, adjust=False).mean().values
    reg = np.where((st == 1) & (c >= e10), 1, np.where((st == -1) & (c <= e10), -1, 0)).astype(np.int8)
    hi = f.high.rolling(21).max().shift(1).values
    lo = f.low.rolling(21).min().shift(1).values
    atr = _atr(f, 14).values
    ax = adx(f)
    pos = np.zeros(len(c), np.int8)
    cur, stop, best, tp = 0, 0.0, 0.0, None
    for i in range(len(c)):
        if cur == 1:
            best = max(best, c[i])
            stop = max(stop, best - 6.5 * atr[i])
            if c[i] <= stop:
                cur = 0
        elif cur == -1:
            best = min(best, c[i])
            stop = min(stop, best + 3.5 * atr[i])
            if c[i] >= stop or (tp is not None and c[i] <= tp):
                cur = 0
        if cur == 0 and not np.isnan(hi[i]):
            if reg[i] == 1 and ax[i] >= 10 and c[i] > hi[i]:
                cur, stop, best = 1, c[i] - 3 * atr[i], c[i]
            elif reg[i] == -1 and ax[i] >= 15 and c[i] < lo[i]:
                cur, stop, best, tp = -1, c[i] + 4 * atr[i], c[i], c[i] - 5 * atr[i]
        pos[i] = cur
    return reg, pos


def main():
    old = pd.read_pickle('out_stage13_test.pkl')
    rows = []
    for a in COINS:
        f = fr(a)
        reg, pos = strategy_states(f)
        for yr in range(2021, 2027):
            lo_, hi_ = pd.Timestamp(f'{yr}-01-01', tz='UTC'), min(pd.Timestamp(f'{yr + 1}-01-01', tz='UTC'), TEST_END)
            w = (f.index >= lo_) & (f.index < hi_) & f.final.values
            if w.sum() < 200 or len(set(f.y.values[w])) < 2:
                continue
            lc = np.log(f.close.values[w])
            r = np.r_[np.diff(lc), 0.0]
            for k, s in (('Video strategy: regime (ST24x4 + EMA10)', reg), ('Video strategy: actual position', pos)):
                fpt, mis, dl, gv, nseg = lag_metrics(s[w], f.y.values[w].astype(np.int8), lc)
                p = (s[w] == 1).astype(float)
                pnl = p * r - 0.001 * np.abs(np.diff(p, prepend=0.0))
                rows.append(dict(group='held-out coin' if a in HOLDOUT else 'training coin', asset=a, year=yr, clf=k,
                                 fpt=fpt, miss=mis, delay=dl, giveback=gv, nseg=nseg,
                                 mcc=matthews_corrcoef(f.y.values[w], np.where(s[w] == 1, 1, -1)),
                                 lf_sharpe=pnl.mean() / pnl.std() * np.sqrt(2190) if pnl.std() > 0 else np.nan,
                                 bh_sharpe=r.mean() / r.std() * np.sqrt(2190), time_in=float(np.mean(s[w] != 0))))
    R = pd.concat([old, pd.DataFrame(rows)])
    R.to_pickle('out_strat_lag.pkl')
    keep = ['STAGE-13 MODEL', 'Stage-12 model', 'Crypto model C (walk-forward)', 'SuperTrend',
            'Video strategy: regime (ST24x4 + EMA10)', 'Video strategy: actual position']
    x = R[R.clf.isin(keep)]
    pd.set_option('display.width', 220)
    print(x.groupby('clf')[['miss', 'delay', 'fpt', 'giveback', 'mcc', 'lf_sharpe']].median().loc[keep].round(3).to_string())
    print(x.groupby('clf').time_in.median().dropna().round(2))
    w = x.pivot_table(index=['asset', 'year'], columns='clf', values='miss')
    from scipy.stats import binomtest
    for b in keep[4:]:
        d = (w[b] - w['STAGE-13 MODEL']).dropna(); k = int((d > 0).sum()); n = int((d != 0).sum())
        print('model less missed move than', b, f'{k}/{n}', 'p=%.2g' % binomtest(k, n).pvalue)
    w = x.pivot_table(index=['asset', 'year'], columns='clf', values='mcc')
    for b in keep[4:]:
        d = (w['STAGE-13 MODEL'] - w[b]).dropna(); k = int((d > 0).sum()); n = int((d != 0).sum())
        print('model higher MCC than', b, f'{k}/{n}', 'p=%.2g' % binomtest(k, n).pvalue)


if __name__ == '__main__':
    main()

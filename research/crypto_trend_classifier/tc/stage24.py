"""Stage 24: India long-only procedure on US small caps (PREREG_stage24_us_smallcaps.md).
python -m tc.stage24 build|test|extra"""
import sys
import numpy as np
import pandas as pd
from . import stage17 as s17
from .stage16 import daily_features, mtf_weekly, CD, TZ, book, run_book, alpha, stats
from .stage17 import MAS, roll_beta

MARKET, TAG = 'us_sc', 'us_sc17'
DELIST, LO_RANK, HI_RANK = -0.30, 1001, 3000
END_CUT = pd.Timestamp('2026-06-01', tz=TZ)
CHUNK = 700


def rank_long(s):
    return (s.groupby(level=0).rank(pct=True) - 0.5).astype(np.float32)


def panel():
    d = pd.read_parquet('us/us_prices_f32.parquet')
    d['date'] = d.date.dt.tz_localize(TZ)
    d = d.drop_duplicates(['date', 'symbol'])
    d = d[(d[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    W = {k: d.pivot(index='date', columns='symbol', values=c).sort_index().astype(np.float32)
         for k, c in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('V', 'volume'))}
    del d
    W['DV'] = W['C'] * W['V']
    last = W['C'].apply(lambda s: s.last_valid_index())
    delisted = last[last < END_CUT]
    EX = W['O'].shift(-1)
    for s, L in delisted.items():
        EX.loc[L, s] = W['C'].loc[L, s] * (1 + DELIST)
    W['EX'] = EX
    return W, delisted


def build():
    W, delisted = panel()
    C = W['C']
    r = np.log(C.astype(float)).diff().clip(-1, 1).astype(np.float32)
    hist = C.notna().cumsum()
    elig = (hist >= 252) & C.notna() & W['EX'].notna() & (C >= 5) & (W['DV'] > 0)
    dvm = W['DV'].rolling(30, min_periods=10).median().where(elig)
    rk = dvm.rank(axis=1, ascending=False)
    elig = elig & (rk >= LO_RANK) & (rk <= HI_RANK)
    keep = elig.any()
    keep = keep[keep].index
    t0 = elig.any(axis=1).idxmax() - pd.Timedelta('600D')
    W = {k: v.loc[t0:, keep] for k, v in W.items()}
    C, r, elig, dvm = W['C'], r.loc[t0:, keep], elig.loc[t0:, keep], dvm.loc[t0:, keep]
    print('universe columns', len(keep), 'dates', len(C), 'median names/day', int(elig.sum(axis=1).median()), flush=True)
    mkt = r.where(elig).mean(axis=1).astype(float)
    E = np.log(W['EX'].astype(float))
    fwd = E.shift(-21) - E
    mfwd = fwd.where(elig).mean(axis=1)
    del fwd
    lags = (1, 3, 7, 14, 30, 60, 90, 180, 252)
    cols, r30_list, dates_i, assets_i = {}, [], [], []
    for i in range(0, len(keep), CHUNK):
        cs = keep[i:i + CHUNK]
        P = {k: W[k][cs].astype(float) for k in ('O', 'H', 'L', 'C', 'DV')}
        e = elig[cs]
        F = daily_features(P, mkt, lags)
        F.update(mtf_weekly(P, P['C'].index))
        Cc, Hc = P['C'], P['H']
        rc = np.log(Cc).diff().clip(-1, 1)
        for L in MAS:
            F[f'ma_{L}'] = np.log(Cc / Cc.rolling(L, min_periods=max(2, L // 2)).mean())
        F['hi52'] = np.log(Cc / Hc.rolling(252, min_periods=126).max())
        b252 = roll_beta(rc, mkt, 252)
        res = rc - b252.mul(mkt, axis=0)
        rs = res.rolling(252, min_periods=126).std()
        F['resmom_12_1'] = res.shift(21).rolling(231, min_periods=120).sum() / rs
        F['resmom_6_1'] = res.shift(21).rolling(105, min_periods=60).sum() / rs
        F['ivol_60'] = res.rolling(60, min_periods=30).std()
        Ec = E[cs]
        fw = Ec.shift(-21) - Ec
        F['y21r'] = (fw - b252.mul(mfwd, axis=0)).where(fw.notna())
        m = e.values
        di, ci = np.nonzero(m)
        dates_i.append(di.astype(np.int32))
        assets_i.append((ci + i).astype(np.int32))
        for k in list(F):
            cols.setdefault(k, []).append(F.pop(k).values[m].astype(np.float32))   # values only, one shared index
        r30 = np.log(Cc) - np.log(Cc.shift(30))
        r30_list.append(r30.values[m].astype(np.float32))
        del P, F
        print('chunk', i, flush=True)
    di = np.concatenate(dates_i)
    ai = np.concatenate(assets_i)
    order = np.lexsort((ai, di))
    di, ai = di[order], ai[order]
    idx = pd.MultiIndex.from_arrays([C.index[di], keep[ai]], names=['date', 'asset'])
    X = pd.DataFrame(index=idx)
    for k in list(cols):
        v = np.concatenate(cols.pop(k))[order]
        X[k] = (pd.Series(v).groupby(di).rank(pct=True) - 0.5).values.astype(np.float32)
    r30 = pd.Series(np.concatenate(r30_list)[order], index=C.index[di])
    lm = mkt.fillna(0).cumsum()
    ctx = pd.DataFrame({f'mkt_ret_{n}': lm - lm.shift(n) for n in (7, 30, 90)})
    ctx['lead_ret_30'] = ctx['mkt_ret_30']
    ctx['breadth_30'] = (r30 > 0).where(r30.notna()).groupby(level=0).mean()
    ctx['disp_30'] = r30.groupby(level=0).std()
    ctx['mkt_ret_504'] = lm - lm.shift(504)
    ctx['mkt_vol_126'] = mkt.rolling(126).std()
    X = X.join(ctx.astype(np.float32), on='date')
    X.to_pickle(f'{CD}/{TAG}_X.pkl')
    trend = X[[f'ma_{L}' for L in MAS]].mean(axis=1).unstack()
    trend.astype(np.float32).to_pickle(f'{CD}/{TAG}_trend.pkl')
    rex = (E.shift(-1) - E).clip(-1, 1)
    pd.to_pickle(dict(rex=rex.astype(np.float32), elig=elig, r=r, dv=dvm.astype(np.float32), dvrank=None,
                      C=C.astype(np.float32), vol60=r.rolling(60, min_periods=30).std().astype(np.float32),
                      delisted=delisted[delisted.index.isin(keep)]), f'{CD}/{MARKET}_P.pkl')
    print('X', X.shape, X.y21r.notna().sum(), flush=True)


def setup():
    s17.MARKET, s17.COST, s17.LONG_ONLY_PRIMARY = MARKET, 0.002, True
    s17.TAG, s17.FROZEN, s17.OUT = TAG, 'FROZEN_stage18.json', 'stage24_us_sc'
    s17.T0 = pd.Timestamp('2011-01-01', tz=TZ)
    s17.T1 = pd.Timestamp('2026-09-26', tz=TZ)


def test():
    setup()
    s17.test()


def extra():
    setup()
    X, D = s17.load()
    F = s17.controls(D)
    S = pd.read_pickle(f'{CD}/{TAG}_scores.pkl')
    dl = D['delisted']
    Sw, rex, ls, to = s17.books(S, D, s17.T0 - pd.Timedelta('400D'), s17.T1)
    rows = []

    def add(name, x):
        x = x.loc[s17.T0:]
        a, t, _ = alpha(x, F.loc[x.index])
        rows.append(dict(test=name, alpha_ann=a * 252, t=t, **stats(x, 252)))

    add('long-only (primary, 20bp, delist -30%)', run_book(book(Sw, long_only=True), rex, s17.COST, s17.HOLD)[0])
    add('long-only, 40bp costs', run_book(book(Sw, long_only=True), rex, 0.004, s17.HOLD)[0])
    C = D['C']
    for dr in (0.0, -1.0):
        rx = rex.copy()
        for s, L in dl.items():
            if s in rx.columns and L in rx.index:
                j = rx.index.get_loc(L)
                if j > 0:
                    rx.iloc[j - 1, rx.columns.get_loc(s)] = rx.iloc[j - 1, rx.columns.get_loc(s)] / (1 + DELIST) * (1 + dr) \
                        if np.isfinite(rx.iloc[j - 1, rx.columns.get_loc(s)]) else np.nan
        add(f'long-only, delisting {int(dr * 100)}%', run_book(book(Sw, long_only=True), rx, s17.COST, s17.HOLD)[0])
    surv = [c for c in Sw.columns if c not in set(dl.index)]
    add('long-only, survivors only', run_book(book(Sw[surv], long_only=True), rex[surv], s17.COST, s17.HOLD)[0])
    dvr = D['dv'].loc[rex.index].where(Sw.notna()).rank(axis=1, ascending=False)
    add('long-only, more liquid half', run_book(book(Sw.where(dvr <= 1000), long_only=True), rex, s17.COST, s17.HOLD)[0])
    add('long-only, less liquid half', run_book(book(Sw.where(dvr > 1000), long_only=True), rex, s17.COST, s17.HOLD)[0])
    R = pd.DataFrame(rows)
    R.to_csv('results_stage24_us_sc_extra.csv', index=False)
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())
    w = book(Sw, long_only=True).rolling(s17.HOLD, min_periods=1).mean()
    held = (w.loc[s17.T0:] > 0).any()
    print('stocks ever held:', int(held.sum()), ' later delisted:', int(held[held].index.isin(dl.index).sum()))


if __name__ == '__main__':
    dict(build=build, test=test, extra=extra)[sys.argv[1]]()

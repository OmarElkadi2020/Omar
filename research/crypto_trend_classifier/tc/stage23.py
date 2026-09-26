"""Stage 23: stage-18 India procedure on survivorship-free NSE bhavcopy data (PREREG_stage23).
python -m tc.stage23 panel|build|test|extra"""
import json
import sys
import numpy as np
import pandas as pd
from . import stage16, stage17 as s17
from .stage16 import CD, TZ, book, run_book, alpha, stats

DELIST = -0.30
MARKET = 'india_sf'


def link_symbols(B):
    """map renamed symbols onto one chain id: A ends, B starts within 5 trading days, B.PREV_CLOSE ~ A.CLOSE."""
    days = np.sort(B.date.unique())
    pos = {d: i for i, d in enumerate(days)}
    g = B.groupby('SYMBOL')
    first = g.head(1).set_index('SYMBOL')
    last = g.tail(1).set_index('SYMBOL')
    end = days[-1]
    starts = first.reset_index()
    starts['i'] = starts.date.map(pos)
    parent = {}
    for a, row in last.iterrows():
        if row.date >= days[-6]:
            continue
        ia = pos[row.date]
        cand = starts[(starts.i > ia) & (starts.i <= ia + 5) & (starts.SYMBOL != a)]
        if len(cand) == 0 or not row.CLOSE_PRICE > 0:
            continue
        ratio = (cand.PREV_CLOSE / row.CLOSE_PRICE - 1).abs()
        k = ratio.idxmin()
        if ratio[k] <= 0.005 and cand.loc[k, 'SYMBOL'] not in parent.values():
            parent[a] = cand.loc[k, 'SYMBOL']
    # follow chains to the final symbol
    def root(s):
        seen = set()
        while s in parent and s not in seen:
            seen.add(s)
            s = parent[s]
        return s
    return {a: root(a) for a in parent}, end


def panel():
    B = pd.read_parquet('bhav_all.parquet')
    B['date'] = pd.to_datetime(B.date).dt.tz_localize(TZ)
    ren, end = link_symbols(B)
    B['chain'] = B.SYMBOL.map(lambda s: ren.get(s, s))
    B = B.sort_values(['chain', 'date', 'pri']).drop_duplicates(['chain', 'date'], keep='last')
    print('renames linked:', len(ren), flush=True)
    r = (B.CLOSE_PRICE / B.PREV_CLOSE).where((B.PREV_CLOSE > 0) & (B.CLOSE_PRICE > 0))
    # bhavcopy PREV_CLOSE is NOT adjusted on ex-dates -> apply bonus / split factors from the NSE corporate-action file
    ca = pd.read_parquet('ca.parquet')
    ca = ca[ca.Action_Type.isin(['BONUS', 'SPLIT']) & ~ca.Purpose.str.contains('Ncrps|Preference|NCRPS', na=False)]
    ab = ca.Ratio_or_Amount.str.extract(r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)').astype(float)
    ca = ca.assign(a=ab[0], b=ab[1]).dropna(subset=['a', 'b'])
    ca = ca[(ca.a > 0) & (ca.b > 0)]
    ca['k'] = np.where(ca.Action_Type == 'BONUS', (ca.a + ca.b) / ca.b, ca.a / ca.b)
    ca['date'] = pd.to_datetime(ca.Ex_Date, errors='coerce').dt.tz_localize(TZ)
    ca['chain'] = ca.Symbol.map(lambda s: ren.get(s, s))
    ca = ca.dropna(subset=['date']).groupby(['chain', 'date']).k.prod().reset_index()
    B = B.reset_index(drop=True)
    r = r.reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(B[['chain', 'date']])
    kk = pd.Series(1.0, index=idx)
    # an ex-date that is not a trading day of the stock applies to its next trading day
    for ch, g in ca.groupby('chain'):
        if ch not in B.chain.values:
            continue
        dts = B.loc[B.chain == ch, 'date'].values
        for d, k in zip(g.date.values, g.k.values):
            j = np.searchsorted(dts, d)
            if j < len(dts):
                kk.loc[(ch, pd.Timestamp(dts[j]).tz_localize(TZ) if pd.Timestamp(dts[j]).tzinfo is None else pd.Timestamp(dts[j]))] *= k
    r = r * kk.values
    print('corporate actions applied:', int((kk != 1).sum()), flush=True)
    # price bands: a >35 % one-day move is an unrecorded corporate action -> keep only the intraday open->close move
    firstday = (B.groupby('chain').cumcount() == 0).values
    bad = ((r - 1).abs() > 0.35).values & ~firstday
    intraday = (B.CLOSE_PRICE / B.OPEN_PRICE).where(B.OPEN_PRICE > 0)
    r = r.where(~bad, intraday.values)
    print('unrecorded gaps neutralised:', int(bad.sum()), flush=True)
    lr = np.log(r).clip(-1, 1)
    first = B.groupby('chain').cumcount() == 0
    lr[first] = 0.0
    B['cum'] = lr.fillna(0).groupby(B.chain).cumsum()
    # anchor the adjusted series to the raw close on each chain's last day
    B['f'] = np.exp(B.cum - B.groupby('chain').cum.transform('last')) * B.groupby('chain').CLOSE_PRICE.transform('last') / B.CLOSE_PRICE
    P = {}
    for k, c in (('O', 'OPEN_PRICE'), ('H', 'HIGH_PRICE'), ('L', 'LOW_PRICE'), ('C', 'CLOSE_PRICE')):
        P[k] = (B[c] * B.f).where(B[c] > 0)
    tmp = pd.DataFrame({k: v for k, v in P.items()})
    tmp['RAW'] = B.CLOSE_PRICE.where(B.CLOSE_PRICE > 0)
    tmp['V'] = B.TTL_TRD_QNTY.astype(float)
    tmp['DV'] = B.CLOSE_PRICE * B.TTL_TRD_QNTY
    tmp['date'], tmp['chain'] = B.date, B.chain          # keep tz (.values drops it)
    W = {k: tmp.pivot(index='date', columns='chain', values=k).sort_index() for k in ('O', 'H', 'L', 'C', 'RAW', 'V', 'DV')}
    lastd = B.groupby('chain').date.max()
    delisted = lastd[lastd < B.date.max() - pd.Timedelta('10D')]
    O, C = W['O'], W['C']
    EX = O.shift(-1)                      # execution for signal D: next open (adjusted)
    # delisting: the position held into the last trading day L exits at close(L) * (1 + DELIST)
    for s, L in delisted.items():
        EX.loc[L, s] = C.loc[L, s] * (1 + DELIST)
    W['EX'] = EX
    W['A'] = C
    pd.to_pickle(dict(P=W, delisted=delisted, renames=ren), f'{CD}/{MARKET}_raw.pkl')
    print('panel', C.shape, 'delisted chains', len(delisted), flush=True)


def load_sf():
    return pd.read_pickle(f'{CD}/{MARKET}_raw.pkl')['P']


def build():
    stage16.CFG[MARKET] = dict(stage16.CFG['india'])
    stage16.load_india = load_sf                      # same builder, survivorship-free panel
    stage16.build_market_alias = MARKET
    stage16.build(MARKET)
    s17.MARKET, s17.TAG = MARKET, 'india_sf17'
    s17.load_india = load_sf
    s17.build()


def setup():
    s17.MARKET, s17.COST, s17.LONG_ONLY_PRIMARY = MARKET, 0.0015, True
    s17.TAG, s17.FROZEN, s17.OUT = 'india_sf17', 'FROZEN_stage18.json', 'stage23_india_sf'
    s17.T0 = pd.Timestamp('2014-01-01', tz=TZ)
    s17.T1 = pd.Timestamp('2026-09-26', tz=TZ)


def test():
    setup()
    s17.test()


def extra():
    """secondary: delisting sensitivity, survivor-only subset, liquidity subsets, delisted holdings."""
    setup()
    X, D = s17.load()
    F = s17.controls(D)
    S = pd.read_pickle(f'{CD}/india_sf17_scores.pkl')
    raw = pd.read_pickle(f'{CD}/{MARKET}_raw.pkl')
    Wr, dl = raw['P'], raw['delisted']
    Sw, rex, ls, to = s17.books(S, D, s17.T0 - pd.Timedelta('400D'), s17.T1)
    rows = []

    def add(name, x, extra=None):
        x = x.loc[s17.T0:]
        a, t, _ = alpha(x, F.loc[x.index])
        rows.append(dict(test=name, alpha_ann=a * 252, t=t, **stats(x, 252), **(extra or {})))

    lo = run_book(book(Sw, long_only=True), rex, s17.COST, s17.HOLD)[0]
    add('long-only (primary, delist -30%)', lo)
    for dr in (0.0, -1.0):
        rx = rex.copy()
        for s, L in dl.items():
            if s in rx.columns and L in rx.index:
                prev = rx.index[rx.index.get_loc(L) - 1] if rx.index.get_loc(L) > 0 else None
                if prev is not None and pd.notna(Wr['O'].loc[L, s]) and Wr['O'].loc[L, s] > 0:
                    rx.loc[prev, s] = Wr['C'].loc[L, s] * (1 + dr) / Wr['O'].loc[L, s] - 1
        add(f'long-only, delisting return {int(dr * 100)}%', run_book(book(Sw, long_only=True), rx, s17.COST, s17.HOLD)[0])
    surv = [c for c in Sw.columns if c not in set(dl.index)]
    add('long-only, survivors only (bias check)', run_book(book(Sw[surv], long_only=True), rex[surv], s17.COST, s17.HOLD)[0])
    dv = D['dv'].loc[rex.index]
    for k in (300, 200, 100):
        Sk = Sw.where(dv.where(Sw.notna()).rank(axis=1, ascending=False) <= k)
        add(f'long-only, top-{k} liquid', run_book(book(Sk, long_only=True), rex, s17.COST, s17.HOLD)[0])
    add('long-short (unscaled)', ls)
    w = book(Sw, long_only=True).rolling(s17.HOLD, min_periods=1).mean()
    held_delisted = [s for s in dl.index if s in w.columns and (w[s].loc[s17.T0:dl[s]] > 0).any()]
    ever = int((w.loc[s17.T0:] > 0).any().sum())
    R = pd.DataFrame(rows)
    R.to_csv('results_stage23_india_sf_extra.csv', index=False)
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())
    print(f'stocks ever held long: {ever}; of them later delisted: {len(held_delisted)}')
    print('delisted chains in universe period:', int(((dl >= s17.T0)).sum()))


if __name__ == '__main__':
    dict(panel=panel, build=build, test=test, extra=extra)[sys.argv[1]]()

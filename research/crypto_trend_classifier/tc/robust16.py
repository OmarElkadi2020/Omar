"""Stage 16 robustness (secondary, post-hoc, labelled as such): realistic shorting, funding, costs, universe, concentration."""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from .stage16 import load, book, run_book, alpha, stats, CFG, CD, TZ, factors


def funding_panel(index):
    F = {}
    for f in glob.glob('bnc/fund/*.pkl'):
        d = pd.read_pickle(f)
        t = pd.to_datetime(d.calc_time.astype('int64'), unit='ms', utc=True)
        s = pd.Series(d.last_funding_rate.astype(float).values, index=t).sort_index()
        s = s[~s.index.duplicated()]
        # funding paid during signal-day D's holding window (04:00 D+1 -> 04:00 D+2), approximated by UTC day D+1
        F[os.path.basename(f)[:-4]] = s.resample('1D').sum().shift(-1)
    fund = pd.DataFrame(F).reindex(index)
    listed = pd.DataFrame({k: v.reindex(index).notna().cumsum() > 0 for k, v in F.items()}).reindex(index).fillna(False)
    listed = listed.shift(1, fill_value=False)       # perp must already be trading the day before the signal
    return fund, listed


def main(market='crypto'):
    cfg = CFG[market]
    p = json.load(open(f'FROZEN_stage16_{market}.json'))['params']
    X, D = load(market)
    t0, t1 = (pd.Timestamp(x, tz=TZ) for x in cfg['test'])
    S = pd.read_pickle(f'{CD}/{market}_scores.pkl').unstack()
    rex = D['rex'].loc[t0:t1]
    S = S.reindex(index=rex.index)
    F = factors(D, market).loc[rex.index]
    bpy, hold = cfg['bpy'], p['hold']
    rows = []

    def add(name, ret):
        ret = ret.loc[S.notna().sum(axis=1).gt(0).idxmax():]
        a, t, _ = alpha(ret, F.loc[ret.index])
        rows.append(dict(variant=name, alpha_ann=a * bpy, alpha_t=t, **stats(ret, bpy)))
        print(rows[-1], flush=True)

    base, _ = run_book(book(S), rex, cfg['cost'], hold)
    add('baseline (as reported, 10bp)', base)
    for c in ((0.002, 0.003) if market == 'crypto' else (0.001, 0.0015)):
        add(f'costs {c * 1e4:g}bp', run_book(book(S), rex, c, hold)[0])
    if market == 'stocks':
        return stocks_extra(S, rex, F, cfg, hold, base, add, rows, D)
    fund, listed = funding_panel(rex.index)
    # long leg on spot; short leg only in coins with a live USDT-M perpetual, paying/receiving real funding
    rk = S.rank(axis=1, pct=True)
    n = S.notna().sum(axis=1)
    top = rk > 0.8
    bot = (rk <= 0.2) & listed.reindex(columns=S.columns, fill_value=False)
    wl = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    ws = bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = (wl - ws).mul((n >= 10).astype(float), axis=0)
    wh = w.rolling(hold, min_periods=1).mean()
    ret, to = run_book(w, rex, cfg['cost'], hold)
    fr = fund.reindex(columns=S.columns).fillna(0)
    carry = -(wh.clip(upper=0) * fr).sum(axis=1)          # short pays funding when rate < 0, receives when > 0
    add('shorts only via live perps + real funding (10bp)', ret + carry)
    add('  same, 20bp costs', run_book(w, rex, 0.002, hold)[0] + carry)
    add('  same, shorts always PAY |funding|', ret - (wh.clip(upper=0).abs() * fr.abs()).sum(axis=1))
    # legs
    mkt = F['MKT']
    add('long leg minus EW market', run_book(book(S, long_only=True), rex, cfg['cost'], hold)[0] - mkt)
    sb = -((rk <= 0.2)).div((rk <= 0.2).sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    add('EW market minus short leg', mkt + run_book(sb, rex, cfg['cost'], hold)[0])
    # top-50 liquid only
    dv = D['dv'].loc[rex.index]
    S50 = S.where(dv.where(S.notna()).rank(axis=1, ascending=False) <= 50)
    add('top-50 by dollar volume only', run_book(book(S50), rex, cfg['cost'], hold)[0])
    # concentration: remove the 5 assets with the largest total contribution
    wd = w.rolling(hold, min_periods=1).mean()
    contrib = (wd * rex.fillna(0)).sum().sort_values()
    drop = list(contrib.index[-5:])
    add(f'drop top-5 contributors {drop}', run_book(book(S.drop(columns=drop)), rex.drop(columns=drop), cfg['cost'], hold)[0])
    q = base.quantile(0.99)
    add('best 1% of days removed', base[base < q])
    R = pd.DataFrame(rows)
    R.to_csv(f'results_stage16_{market}_robustness.csv', index=False)
    print(R.round(3).to_string())


def stocks_extra(S, rex, F, cfg, hold, base, add, rows, D):
    from .stage10 import splits
    wsh = -book(S).clip(upper=0).rolling(hold, min_periods=1).mean().sum(axis=1)
    add('borrow fee 1%/yr on shorts', base - wsh * 0.01 / 252)
    add('long leg minus EW market', run_book(book(S, long_only=True), rex, cfg['cost'], hold)[0] - F['MKT'])
    rk = S.rank(axis=1, pct=True)
    sb = -((rk <= 0.2)).div((rk <= 0.2).sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    add('EW market minus short leg', F['MKT'] + run_book(sb, rex, cfg['cost'], hold)[0])
    A = [t for t in splits()['A'] if t in S.columns]
    add('top-50 large caps only (set A)', run_book(book(S[A]), rex[A], cfg['cost'], hold)[0])
    dv = D['dv'].loc[rex.index]
    S200 = S.where(dv.where(S.notna()).rank(axis=1, ascending=False) <= 200)
    add('top-200 by dollar volume only', run_book(book(S200), rex, cfg['cost'], hold)[0])
    add('best 1% of days removed', base[base < base.quantile(0.99)])
    R = pd.DataFrame(rows)
    R.to_csv('results_stage16_stocks_robustness.csv', index=False)
    print(R.round(3).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:])

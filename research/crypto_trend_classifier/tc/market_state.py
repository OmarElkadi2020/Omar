"""Market-state snapshot (descriptive, not a forecast). Reproduces MARKET_STATE_2026-10-06.md from repo data.
Inputs: data/universe_4h, data/binance_1h, data/coinalyze (+ COINALYZE_API_KEY for the long/short-ratio history).
python -m tc.market_state"""
import json
import os
import time
import urllib.request
import numpy as np
import pandas as pd
from .evaluate import supertrend

CH = {'4h': (48, 5.0), '8h': (30, 4.0), '12h': (20, 4.0), '1D': (48, 6.0)}   # stage-27 choices


def bars(sym):
    d = pd.read_parquet(f'data/universe_4h/{sym}.parquet').astype(float)
    return d[d.close > 0]


def agg(d, tf):
    if tf == '4h':
        return d
    return d.resample(tf, origin='epoch').agg(dict(open='first', high='max', low='min', close='last',
                                                   volume='sum')).dropna()


def trend_table(syms=('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT')):
    rows = []
    b = agg(bars('BTCUSDT'), '1D').close
    for s in syms:
        d = bars(s)
        c = agg(d, '1D').close
        r = np.log(c).diff()
        row = dict(asset=s, close=c.iloc[-1], **{f'st_{tf}': 'UP' if supertrend(agg(d, tf), *CH[tf])[-1] == 1 else 'DOWN'
                                                for tf in CH},
                   from_ath=c.iloc[-1] / c.max() - 1, from_1y_high=c.iloc[-1] / c[-365:].max() - 1,
                   vol30=r[-30:].std() * np.sqrt(365), vol30_1y_median=r.rolling(30).std()[-365:].median() * np.sqrt(365))
        for h in (7, 30, 90, 365):
            row[f'ret_{h}d'] = c.iloc[-1] / c.iloc[-h - 1] - 1
            bb = b.reindex(c.index)
            row[f'vs_btc_{h}d'] = (c.iloc[-1] / bb.iloc[-1]) / (c.iloc[-h - 1] / bb.iloc[-h - 1]) - 1
        rows.append(row)
    return pd.DataFrame(rows).set_index('asset')


def breadth(universe):
    hist = {}
    for s in universe:
        a = agg(bars(s), '1D')
        if len(a) >= 120:
            hist[s] = pd.Series(supertrend(a, *CH['1D']), a.index)
    H = pd.DataFrame(hist)
    return ((H == 1).mean(axis=1) * 100).loc['2026-01-01':]


def waves(sym):
    """daily SuperTrend(48,6) up-waves: length, gain, and the max pullback in the first 46 days
    (low vs the highest high BEFORE that day, and on closes)."""
    a = agg(bars(sym), '1D')
    st = pd.Series(supertrend(a, *CH['1D']), a.index)
    ch = np.flatnonzero(np.diff(st.values) != 0) + 1
    rows = []
    for i, j in zip(np.r_[0, ch], np.r_[ch, len(st)]):
        if st.iloc[i] != 1 or i < 60:
            continue
        w = a.iloc[i:min(j, i + 47)]
        rows.append(dict(start=st.index[i].date(), days=j - i, gain_at_end=a.close.iloc[j - 1] / a.close.iloc[i] - 1,
                         max_gain=a.close.iloc[i:j].max() / a.close.iloc[i] - 1,
                         pullback_low=(w.low / w.high.cummax().shift(1) - 1).min(),
                         pullback_close=(w.close / w.close.cummax() - 1).min(), ongoing=j == len(st)))
    return pd.DataFrame(rows)


def lsr_history():
    f = 'data/coinalyze/lsr_daily.parquet'
    if os.path.exists(f) and 'COINALYZE_API_KEY' not in os.environ:
        return pd.read_parquet(f)
    rows, lo, end = [], pd.Timestamp('2020-01-01'), pd.Timestamp.now()
    while lo < end:
        hi = min(lo + pd.Timedelta(days=900), end)
        req = urllib.request.Request(
            'https://api.coinalyze.net/v1/long-short-ratio-history?symbols=BTCUSDT_PERP.A,ETHUSDT_PERP.A&interval=daily'
            f'&from={int(lo.timestamp())}&to={int(hi.timestamp())}', headers={'api_key': os.environ['COINALYZE_API_KEY']})
        for s in json.load(urllib.request.urlopen(req, timeout=60)):
            rows += [dict(c=s['symbol'][:3], t=h['t'], r=h['r']) for h in s['history']]
        lo = hi
        time.sleep(2)
    L = pd.DataFrame(rows).drop_duplicates(['c', 't'])
    L['t'] = pd.to_datetime(L.t, unit='s', utc=True).dt.normalize()
    L = L.pivot(index='t', columns='c', values='r')
    L.to_parquet(f)
    return L


def lsr_study(L):
    out = {}
    for c in ('BTC', 'ETH'):
        px = pd.read_parquet(f'data/binance_1h/{c}.parquet').close.resample('1D').last()
        pct = L[c].dropna().rolling(365, min_periods=180).apply(lambda x: (x <= x[-1]).mean(), raw=True)
        df = pd.DataFrame({'pct': pct}).join(px.rename('p'), how='inner').dropna()
        for h in (7, 30, 90):
            df[f'f{h}'] = np.log(px.shift(-h) / px).reindex(df.index)
        df = df.dropna()
        for h in (7, 30, 90):
            g = df.groupby(pd.cut(df.pct, [0, .2, .4, .6, .8, 1.0], include_lowest=True), observed=False)[f'f{h}'].mean()
            out[f'{c} {h}d mean log-return by L/S quintile'] = [round(x * 100, 1) for x in g]
    return out


def main():
    pd.set_option('display.width', 220)
    T = trend_table()
    print(T.round(3).T.to_string())
    U = list(pd.read_csv('results_live16_rank.csv', index_col=0).index)
    b = breadth(U)
    print('\ndaily-ST breadth (current top-100), month ends:', b.resample('ME').last().round(0).tolist())
    for s in ('BTCUSDT', 'ETHUSDT'):
        W = waves(s)
        print(f'\n{s} waves'); print(W.round(3).to_string(index=False))
    P = pd.read_parquet('data/coinalyze/btc_eth_positioning_daily_2025-09_2026-10.parquet')
    for c in ('BTC', 'ETH'):
        oc = P[f'oi_coin_c_{c}'].dropna()
        print(f'\n{c}: OI in coins since 2026-08-20 {oc.iloc[-1] / oc.loc["2026-08-20"] - 1:+.1%} | funding 7d mean '
              f'{P[f"fund_c_{c}"][-7:].mean():.4f}%/8h | L/S {P[f"lsr_r_{c}"].iloc[-1]:.2f} | liquidations since 08-20 '
              f'long ${P[f"liq_l_{c}"].loc["2026-08-20":].sum() / 1e6:.0f}m short ${P[f"liq_s_{c}"].loc["2026-08-20":].sum() / 1e6:.0f}m')
    try:
        print('\n', json.dumps(lsr_study(lsr_history()), indent=1))
    except Exception as e:
        print('L/S study skipped:', e)


if __name__ == '__main__':
    main()

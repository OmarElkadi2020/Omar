"""Hourly anatomy of a market move (descriptive). Needs COINALYZE_API_KEY.
Saves Binance-perp hourly OHLCV (incl. taker-buy volume), open interest (coins), liquidations (USD) and funding to
data/coinalyze/intraday_<tag>.parquet, prints the hour-by-hour table around the move, the current SuperTrend levels,
and BTC liquidity / price impact by hour of day (from data/binance_1h).
python -m tc.intraday_snapshot 2026-10-07 [days_back]"""
import json
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from .features import _atr

COINS = ('BTC', 'ETH', 'SOL', 'LINK')


def get(path):
    for _ in range(6):
        try:
            req = urllib.request.Request('https://api.coinalyze.net/v1/' + path,
                                         headers={'api_key': os.environ['COINALYZE_API_KEY']})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(float(e.headers.get('Retry-After', 10)) + 1)
    raise RuntimeError('rate limited')


def fetch(days):
    now = int(time.time())
    fr = now - days * 86400
    syms = ','.join(f'{c}USDT_PERP.A' for c in COINS)
    cols = {}
    for ep, extra, fields, pre in (('ohlcv-history', '', ('o', 'h', 'l', 'c', 'v', 'bv'), 'ohlcv'),
                                   ('open-interest-history', '', ('c',), 'oi'),
                                   ('liquidation-history', '&convert_to_usd=true', ('l', 's'), 'liquidation'),
                                   ('funding-rate-history', '', ('c',), 'funding')):
        for s in get(f'{ep}?symbols={syms}&interval=1hour&from={fr}&to={now}{extra}'):
            d = pd.DataFrame(s['history'])
            d.index = pd.to_datetime(d.t, unit='s', utc=True)
            c = s['symbol'].split('USDT')[0]
            for f in fields:
                cols[f'{pre}_{f}_{c}'] = d[f]
        time.sleep(2)
    D = pd.DataFrame(cols)
    D.index.name = 'hour'
    return D


def st_line(f, p, m):
    a = _atr(f, p).values
    h, l, c = f.high.values, f.low.values, f.close.values
    fub = flb = np.nan
    dd, ln = 1, np.nan
    for i in range(len(c)):
        if np.isnan(a[i]):
            continue
        hl2 = (h[i] + l[i]) / 2
        ub, lb = hl2 + m * a[i], hl2 - m * a[i]
        if np.isnan(fub):
            fub, flb = ub, lb
        else:
            fub = ub if (ub < fub or c[i - 1] > fub) else fub
            flb = lb if (lb > flb or c[i - 1] < flb) else flb
        if dd == 1 and c[i] < flb:
            dd = -1
        elif dd == -1 and c[i] > fub:
            dd = 1
        ln = flb if dd == 1 else fub
    return dd, ln


def levels(D, c):
    """SuperTrend(48,5) on 4h and (48,6) daily: spot history from data/universe_4h, then perp hours."""
    sp = pd.read_parquet(f'data/universe_4h/{c}USDT.parquet').astype(float)[['open', 'high', 'low', 'close']]
    p = D[[f'ohlcv_{k}_{c}' for k in 'ohlc']].dropna()
    p.columns = ['open', 'high', 'low', 'close']
    p4 = p.resample('4h', origin='epoch').agg(dict(open='first', high='max', low='min', close='last')).dropna()
    full = pd.concat([sp[sp.index < p4.index[0]], p4])
    d4, l4 = st_line(full.iloc[:-1], 48, 5.0)                       # last completed 4h bar
    day = full.resample('1D').agg(dict(open='first', high='max', low='min', close='last')).dropna()
    d1, l1 = st_line(day.iloc[:-1], 48, 6.0)
    return dict(st_4h='UP' if d4 == 1 else 'DOWN', line_4h=round(l4, 2), st_1d='UP' if d1 == 1 else 'DOWN',
                line_1d=round(l1, 2), last=p.close.iloc[-1])


def anatomy(D, c, start, end):
    w = D.loc[start:end]
    o = D[f'oi_c_{c}']
    t = pd.DataFrame({'close': w[f'ohlcv_c_{c}'], 'low': w[f'ohlcv_l_{c}'], 'volume': w[f'ohlcv_v_{c}'],
                      'buy_share_%': w[f'ohlcv_bv_{c}'] / w[f'ohlcv_v_{c}'] * 100,
                      'oi_chg_%': o.pct_change().reindex(w.index) * 100,
                      'long_liq_$m': w[f'liquidation_l_{c}'].fillna(0) / 1e6,
                      'short_liq_$m': w[f'liquidation_s_{c}'].fillna(0) / 1e6, 'funding_%': w[f'funding_c_{c}']})
    return t


def hour_of_day_liquidity(since='2025-10-01'):
    d = pd.read_parquet('data/binance_1h/BTC.parquet').loc[since:]
    r = np.log(d.close).diff().abs()
    v = d.quote_volume.groupby(d.index.hour).median()
    imp = (r / d.quote_volume).groupby(d.index.hour).median()
    return pd.DataFrame({'volume_vs_avg': (v / v.mean()).round(2), 'impact_vs_avg': (imp / imp.mean()).round(2)})


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    D = fetch(days)
    D.to_parquet(f'data/coinalyze/intraday_{tag}.parquet')
    pd.set_option('display.width', 200)
    day = pd.Timestamp(tag, tz='UTC')
    for c in ('BTC', 'ETH'):
        print(f'\n== {c} ==', levels(D, c))
        print(anatomy(D, c, day - pd.Timedelta(hours=4), day + pd.Timedelta(hours=23)).round(3).to_string())
    print('\nBTC liquidity by hour of day (UTC):')
    print(hour_of_day_liquidity().T.to_string())


if __name__ == '__main__':
    main()

"""Hourly risk watch (descriptive alerts, not advice). Needs COINALYZE_API_KEY; COIN_MARKET_CAP_API_KEY and
BYkaranteli_API_KEY are optional extras. Prints one JSON line and appends it to data/watch_log.jsonl (gitignored).
Alert levels, fixed in advance (stage-27 SuperTrend settings, decisions on CLOSED candles only):
  0 OK       BTC and ETH: 4h and daily SuperTrend up
  1 WARNING  a completed 4h candle of BTC or ETH closed below its 4h SuperTrend(48,5) line
  2 DANGER   BTC 4h is down AND at least two of: ETH 4h down; BTC long liquidations > $100m in the last 6h
             (Binance perp); stablecoin market cap -1% or more vs the previous reading; BTC spot-ETF net outflow
             two days in a row (latest two rows)
  3 EXIT     a completed DAILY candle of BTC closed below its daily SuperTrend(48,6) line (the wave is over)
python -m tc.watch"""
import json
import os
import time
import urllib.request
import numpy as np
import pandas as pd
from .intraday_snapshot import st_line

LOG = 'data/watch_log.jsonl'


def get(path):
    for _ in range(8):
        try:
            req = urllib.request.Request('https://api.coinalyze.net/v1/' + path,
                                         headers={'api_key': os.environ['COINALYZE_API_KEY']})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(float(e.headers.get('Retry-After', 10)) + 1)
    raise RuntimeError('rate limited')


def hourly(days=10):
    now = int(time.time())
    fr = now - days * 86400
    O, L = {}, {}
    syms = 'BTCUSDT_PERP.A,ETHUSDT_PERP.A'
    for s in get(f'ohlcv-history?symbols={syms}&interval=1hour&from={fr}&to={now}'):
        d = pd.DataFrame(s['history'])
        d.index = pd.to_datetime(d.t, unit='s', utc=True)
        O[s['symbol'][:3]] = d.rename(columns=dict(o='open', h='high', l='low', c='close'))
    time.sleep(2)
    for s in get(f'liquidation-history?symbols={syms}&interval=1hour&from={fr}&to={now}&convert_to_usd=true'):
        d = pd.DataFrame(s['history'])
        d.index = pd.to_datetime(d.t, unit='s', utc=True)
        L[s['symbol'][:3]] = d
    return O, L


def trend(c, h):
    sp = pd.read_parquet(f'data/universe_4h/{c}USDT.parquet').astype(float)[['open', 'high', 'low', 'close']]
    p = h[['open', 'high', 'low', 'close']]
    agg = dict(open='first', high='max', low='min', close='last')
    p4 = p.resample('4h', origin='epoch').agg(agg).dropna()
    full = pd.concat([sp[sp.index < p4.index[0]], p4])
    now = pd.Timestamp.utcnow()
    closed4 = full[full.index + pd.Timedelta('4h') <= now]                      # completed 4h candles only
    d4, l4 = st_line(closed4, 48, 5.0)
    day = full.resample('1D').agg(agg).dropna()
    closed1 = day[day.index + pd.Timedelta('1D') <= now]
    d1, l1 = st_line(closed1, 48, 6.0)
    return dict(price=float(p.close.iloc[-1]), st4h='UP' if d4 == 1 else 'DOWN', line4h=round(float(l4), 2),
                last_4h_close=float(closed4.close.iloc[-1]), last_4h_bar=str(closed4.index[-1]),
                st1d='UP' if d1 == 1 else 'DOWN', line1d=round(float(l1), 2))


def stablecoins():
    if 'COIN_MARKET_CAP_API_KEY' not in os.environ:
        return None
    r = urllib.request.Request('https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest',
                               headers={'X-CMC_PRO_API_KEY': os.environ['COIN_MARKET_CAP_API_KEY']})
    g = json.load(urllib.request.urlopen(r, timeout=60))['data']
    return dict(stable_bn=round(g['quote']['USD']['stablecoin_market_cap'] / 1e9, 2), btc_dom=round(g['btc_dominance'], 2))


def etf_btc():
    if 'BYkaranteli_API_KEY' not in os.environ:
        return None
    try:
        from .download_bykaranteli import fetch, merge
        f = merge('etf-flows', fetch('etf-flows'))
    except Exception:
        f = pd.read_csv('data/bykaranteli/etf-flows.csv')
    b = f[f.asset == 'BTC'].sort_values('date').tail(2)
    return dict(dates=list(b.date), flows_m=[round(x / 1e6, 1) for x in b.net_inflow_usd])


def main():
    O, L = hourly()
    T = {c: trend(c, O[c]) for c in ('BTC', 'ETH')}
    liq6 = {c: round(float(L[c].l.iloc[-6:].sum() / 1e6), 1) for c in L}
    prev = None
    if os.path.exists(LOG):
        lines = [x for x in open(LOG) if x.strip()]
        prev = json.loads(lines[-1]) if lines else None
    st = stablecoins()
    et = etf_btc()
    stable_drop = bool(st and prev and prev.get('stable') and
                       st['stable_bn'] / prev['stable']['stable_bn'] - 1 <= -0.01)
    etf_out = bool(et and len(et['flows_m']) == 2 and all(x < 0 for x in et['flows_m']))
    extras = dict(eth_4h_down=T['ETH']['st4h'] == 'DOWN', btc_long_liq_6h_over_100m=liq6.get('BTC', 0) > 100,
                  stablecoins_minus_1pct=stable_drop, btc_etf_outflow_2_days=etf_out)
    if T['BTC']['st1d'] == 'DOWN':
        level, name = 3, 'EXIT'
    elif T['BTC']['st4h'] == 'DOWN' and sum(extras.values()) >= 2:
        level, name = 2, 'DANGER'
    elif T['BTC']['st4h'] == 'DOWN' or T['ETH']['st4h'] == 'DOWN':
        level, name = 1, 'WARNING'
    else:
        level, name = 0, 'OK'
    out = dict(time=str(pd.Timestamp.utcnow().floor('min')), level=level, status=name, trend=T, long_liq_6h_m=liq6,
               stable=st, etf_btc=et, conditions=extras, previous_level=prev.get('level') if prev else None)
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(out, default=str) + '\n')
    print(json.dumps(out, default=str, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

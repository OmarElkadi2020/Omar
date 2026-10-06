"""Coinalyze daily history (Binance USDT perps): liquidations (USD), open interest (USD, close), funding (close).
Needs COINALYZE_API_KEY. Writes data/coinalyze/{liq,oi,funding}.parquet (columns = coins, index = UTC day).
python -m tc.download_coinalyze"""
import json
import os
import time
import urllib.request
import pandas as pd

COINS = 'BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX'.split()
OUT = 'data/coinalyze'
START = pd.Timestamp('2020-01-01', tz='UTC')


def get(path):
    for k in range(10):
        try:
            req = urllib.request.Request('https://api.coinalyze.net/v1/' + path,
                                         headers={'api_key': os.environ['COINALYZE_API_KEY']})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(float(e.headers.get('Retry-After', 10)) + 1)
    raise RuntimeError('rate limited')


def history(ep, fields, extra=''):
    syms = ','.join(f'{c}USDT_PERP.A' for c in COINS)
    end = pd.Timestamp.now(tz='UTC').normalize()
    rows = []
    lo = START
    while lo < end:
        hi = min(lo + pd.Timedelta(days=900), end)
        for s in get(f'{ep}?symbols={syms}&interval=daily&from={int(lo.timestamp())}&to={int(hi.timestamp())}{extra}'):
            c = s['symbol'].split('USDT')[0]
            rows += [dict(coin=c, t=h['t'], **{k: h[k] for k in fields}) for h in s['history']]
        lo = hi
        time.sleep(3)
    d = pd.DataFrame(rows).drop_duplicates(['coin', 't'])
    d['t'] = pd.to_datetime(d.t, unit='s', utc=True)
    return d


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    L = history('liquidation-history', ['l', 's'], '&convert_to_usd=true')
    L.pivot(index='t', columns='coin', values='l').to_parquet(f'{OUT}/liq_long.parquet')
    L.pivot(index='t', columns='coin', values='s').to_parquet(f'{OUT}/liq_short.parquet')
    O = history('open-interest-history', ['c'], '&convert_to_usd=true')
    O.pivot(index='t', columns='coin', values='c').to_parquet(f'{OUT}/oi.parquet')
    F = history('funding-rate-history', ['c'])
    F.pivot(index='t', columns='coin', values='c').to_parquet(f'{OUT}/funding.parquet')
    for n in ('liq_long', 'liq_short', 'oi', 'funding'):
        d = pd.read_parquet(f'{OUT}/{n}.parquet')
        print(n, d.shape, d.index.min().date(), d.index.max().date(), 'first valid per coin:',
              {c: str(d[c].first_valid_index().date()) for c in d.columns})

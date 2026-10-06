"""Download Binance spot 1h klines (with volume and taker-buy flow) from data.binance.vision monthly files and
store one zstd parquet per coin in data/binance_1h/.  python -m tc.download_binance [first YYYY-MM] [last YYYY-MM] | update
Columns: open time (UTC index), open, high, low, close, volume (base), quote_volume, trades, taker_buy_base,
taker_buy_quote. Newer monthly files use microsecond timestamps; both are handled."""
import io
import http.client
import time
import os
import sys
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

COINS = 'BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX DOT LTC BCH ATOM NEAR UNI FIL ETC XLM AAVE'.split()
URL = 'https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip'
DAY_URL = 'https://data.binance.vision/data/spot/daily/klines/{s}/1h/{s}-1h-{m}.zip'
OUT = 'data/binance_1h'
COLS = ['t', 'open', 'high', 'low', 'close', 'volume', 'ct', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore']


def fetch(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (OSError, http.client.HTTPException) as e:
            err = e
        time.sleep(2 ** (k + 1))
    raise err


def month(sym, m):
    raw = fetch((DAY_URL if len(m) == 10 else URL).format(s=sym, m=m))
    if raw is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    d = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    if not str(d.t.iloc[0]).isdigit():          # header row in a few files
        d = d.iloc[1:].astype({'t': 'int64'})
    t = d.t.astype('int64').values
    t = np.where(t > 10 ** 14, t // 1000, t)     # microseconds -> milliseconds
    d.index = pd.to_datetime(t, unit='ms', utc=True)
    return d[COLS[1:6] + COLS[7:11]].astype({'trades': 'int64'}).astype({c: 'float64' for c in COLS[1:6] + COLS[7:8] + COLS[9:11]})


def update(c):
    """append every complete month, then complete days, after the last bar already stored (forward test)."""
    d = pd.read_parquet(f'{OUT}/{c}.parquet')
    last = d.index[-1]
    today = pd.Timestamp.now(tz='UTC').normalize()
    start = (last + pd.Timedelta(hours=1)).normalize()
    months = [p.strftime('%Y-%m') for p in pd.period_range(start, today - pd.Timedelta(days=1), freq='M')
              if p.end_time.tz_localize('UTC') < today]
    parts = [p for p in (month(f'{c}USDT', m) for m in months) if p is not None]
    got_to = max([p.index[-1] for p in parts], default=last)
    days = pd.date_range(max(start, got_to.normalize() + pd.Timedelta(days=1)), today - pd.Timedelta(days=1), freq='D')
    parts += [p for p in (month(f'{c}USDT', x.strftime('%Y-%m-%d')) for x in days) if p is not None]
    if parts:
        n = pd.concat([d] + parts).sort_index()
        n = n[~n.index.duplicated()]
        n.index.name = 'open_time'
        n.to_parquet(f'{OUT}/{c}.parquet', compression='zstd')
        d = n
    return c, len(d), d.index[-1]


def coin(c, months):
    sym = f'{c}USDT'
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda m: month(sym, m), months) if p is not None]
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()]
    d.index.name = 'open_time'
    d.to_parquet(f'{OUT}/{c}.parquet', compression='zstd')
    return c, len(d), d.index[0], d.index[-1]


if __name__ == '__main__' and sys.argv[1:] == ['update']:
    for c in COINS:
        print(*update(c), flush=True)
elif __name__ == '__main__':
    first, last = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('2019-01', '2026-09')
    months = [p.strftime('%Y-%m') for p in pd.period_range(first, last, freq='M')]
    os.makedirs(OUT, exist_ok=True)
    for c in COINS:
        if os.path.exists(f'{OUT}/{c}.parquet'):
            continue
        print(*coin(c, months), flush=True)

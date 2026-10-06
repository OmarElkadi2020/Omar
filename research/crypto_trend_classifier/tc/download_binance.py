"""Download Binance spot 1h klines (with volume and taker-buy flow) from data.binance.vision monthly files and
store one zstd parquet per coin in data/binance_1h/.  python -m tc.download_binance [first YYYY-MM] [last YYYY-MM]
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
    raw = fetch(URL.format(s=sym, m=m))
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


def coin(c, months):
    sym = f'{c}USDT'
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda m: month(sym, m), months) if p is not None]
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()]
    d.index.name = 'open_time'
    d.to_parquet(f'{OUT}/{c}.parquet', compression='zstd')
    return c, len(d), d.index[0], d.index[-1]


if __name__ == '__main__':
    first, last = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('2019-01', '2026-09')
    months = [p.strftime('%Y-%m') for p in pd.period_range(first, last, freq='M')]
    os.makedirs(OUT, exist_ok=True)
    for c in COINS:
        if os.path.exists(f'{OUT}/{c}.parquet'):
            continue
        print(*coin(c, months), flush=True)

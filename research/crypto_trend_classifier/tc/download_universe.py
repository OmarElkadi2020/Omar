"""Every Binance spot *USDT pair in the public archive (incl. delisted), 4h klines, for the stage-16 ranking.
Writes data/universe_4h/<SYMBOL>.parquet (open/high/low/close/volume, float32, index = open time UTC).
Resumable: finished symbols are skipped; `update` appends new months/days to existing files.
python -m tc.download_universe [update]"""
import io
import os
import re
import sys
import time
import zipfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

S3 = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
BASE = 'https://data.binance.vision/'
OUT = 'data/universe_4h'
NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
# leveraged tokens, wrapped duplicates, stablecoins and fiat (stage-16 prereg: excluded a priori)
EXCL = re.compile(r'(UP|DOWN|BULL|BEAR)USDT$')
STABLE = {'USDC', 'BUSD', 'TUSD', 'FDUSD', 'USDP', 'PAX', 'DAI', 'USDS', 'USDSB', 'SUSD', 'UST', 'EUR', 'GBP', 'AUD',
          'AEUR', 'EURI', 'XUSD', 'USD1', 'PYUSD', 'BFUSD', 'RLUSD', 'USDE', 'IDRT', 'BIDR', 'BKRW', 'TRY', 'BRL', 'RUB',
          'NGN', 'UAH', 'ZAR', 'PLN', 'RON', 'ARS', 'JPY', 'MXN', 'COP', 'CZK', 'U', 'USDQ', 'USDF'}
WRAPPED = {'WBTC', 'WBETH', 'BETH'}


def fetch(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.parse.quote(url, safe=':/?&=%'), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:      # network hiccup
            err = e
        time.sleep(2 ** (k + 1))
    raise err


def listing(prefix):
    out, marker = [], ''
    while True:
        x = ET.fromstring(fetch(f'{S3}?delimiter=/&prefix={prefix}&marker={marker}'))
        out += [p.find(NS + 'Prefix').text for p in x.findall(NS + 'CommonPrefixes')]
        out += [c.find(NS + 'Key').text for c in x.findall(NS + 'Contents')]
        if x.find(NS + 'IsTruncated').text != 'true':
            return out
        marker = x.find(NS + 'NextMarker').text if x.find(NS + 'NextMarker') is not None else out[-1]


def symbols():
    syms = [p.rstrip('/').split('/')[-1] for p in listing('data/spot/monthly/klines/')]
    keep = []
    for s in syms:
        if not s.endswith('USDT') or EXCL.search(s):
            continue
        b = s[:-4]
        if b in STABLE or b in WRAPPED:
            continue
        keep.append(s)
    return keep


def parse(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6),
                    names=['t', 'open', 'high', 'low', 'close', 'volume'])
    if not str(d.t.iloc[0]).isdigit():
        d = d.iloc[1:]
    t = d.t.astype('int64').values
    t = np.where(t > 10 ** 14, t // 1000, t)
    d.index = pd.to_datetime(t, unit='ms', utc=True)
    return d[['open', 'high', 'low', 'close', 'volume']].astype('float32')


def one(sym, after=None):
    keys = [k for k in listing(f'data/spot/monthly/klines/{sym}/4h/') if k.endswith('.zip')]
    if after is not None:
        keys = [k for k in keys if k[-11:-4] >= after.strftime('%Y-%m')]
        dkeys = [k for k in listing(f'data/spot/daily/klines/{sym}/4h/') if k.endswith('.zip')]
        last_m = max([k[-11:-4] for k in keys], default=after.strftime('%Y-%m'))
        keys += [k for k in dkeys if k[-14:-7] > last_m]
    parts = [parse(r) for r in (fetch(BASE + k) for k in keys) if r]
    return pd.concat(parts).sort_index() if parts else None


def save(sym, d):
    d = d[~d.index.duplicated(keep='last')]
    d.index.name = 'open_time'
    d.to_parquet(f'{OUT}/{sym}.parquet', compression='zstd')


def job(sym):
    f = f'{OUT}/{sym}.parquet'
    if UPDATE and os.path.exists(f):
        old = pd.read_parquet(f)
        new = one(sym, after=old.index[-1].normalize())
        if new is not None:
            save(sym, pd.concat([old, new]).sort_index())
        return sym
    if os.path.exists(f):
        return sym
    d = one(sym)
    if d is not None and len(d):
        save(sym, d)
    return sym


UPDATE = sys.argv[1:] == ['update']

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    S = symbols()
    print('symbols', len(S), flush=True)
    with ThreadPoolExecutor(12) as ex:
        for i, s in enumerate(ex.map(job, S)):
            if i % 50 == 0:
                print(i, s, flush=True)
    print('done', len(os.listdir(OUT)), flush=True)

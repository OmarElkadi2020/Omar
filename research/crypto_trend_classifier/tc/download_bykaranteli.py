"""ByKaranteli public datasets (needs BYkaranteli_API_KEY, sent in the x-api-key header).
Each dataset only serves a rolling window (etf-flows: last 12 months), so every run MERGES into the stored file
and the repo copy grows into a longer history over time.
python -m tc.download_bykaranteli [dataset ...]   (default: etf-flows etf-issuer-daily liquidations-daily dvol-daily)"""
import io
import os
import sys
import urllib.request
import pandas as pd

OUT = 'data/bykaranteli'
DEFAULT = ('etf-flows', 'etf-issuer-daily', 'liquidations-daily', 'dvol-daily')


def fetch(name):
    req = urllib.request.Request(f'https://bykaranteli.com/api/v1/public/datasets/{name}?format=csv',
                                 headers={'x-api-key': os.environ['BYkaranteli_API_KEY'], 'User-Agent': 'research'})
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()))


def merge(name, new):
    f = f'{OUT}/{name}.csv'
    keys = [c for c in ('date', 'asset', 'issuer', 'ticker', 'symbol', 'venue', 'side') if c in new.columns]
    if os.path.exists(f):
        new = pd.concat([pd.read_csv(f), new]).drop_duplicates(subset=keys or None, keep='last')
    new = new.sort_values(keys or list(new.columns)[:1]).reset_index(drop=True)
    new.to_csv(f, index=False)
    return new


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for n in (sys.argv[1:] or DEFAULT):
        try:
            d = merge(n, fetch(n))
            dc = 'date' if 'date' in d.columns else d.columns[0]
            print(n, d.shape, d[dc].min(), '->', d[dc].max(), list(d.columns)[:10])
        except Exception as e:
            print(n, 'failed:', e)

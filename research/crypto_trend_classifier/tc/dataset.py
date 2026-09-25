"""Assemble per-(asset, timeframe) frames: OHLC, causal features (+ BTC context), full-hindsight oracle labels."""
import os
import numpy as np
import pandas as pd
from .features import build, join_by_close
from .labels import oracle_labels

COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT', 'DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']
BTC_CTX = ['mom_24', 'mom_96', 'mom_384', 'ema_dist_50', 'ema_dist_200', 'ema_slope_200', 'tstat_96', 'tstat_384',
           'st_24_4.0', 'dc_1.0', 'dc_2.0', 'dc_ov_1.0', 'er_96', 'h4_dc_1.0', 'h4_mom_96', 'h24_dc_1.0', 'h24_mom_96']
CACHE = 'cache'


def invert(d):
    """upside-down copy of a market: price -> 1/price. Up-trends become down-trends exactly (oracle labels
    flip sign), so training on both removes any bull-market prior. Used for TRAINING only."""
    o = d.copy()
    o['open'], o['close'] = 1 / d['open'], 1 / d['close']
    o['high'], o['low'] = 1 / d['low'], 1 / d['high']
    return o


def load_1h(name):
    if name.endswith('_INV'):
        return invert(load_1h(name[:-4]))
    if name == 'BTC_BITSTAMP':
        d = pd.read_pickle('data/BTCUSD.pkl')  # Bitstamp, used only before Binance BTC starts
        d = d[d.index < '2017-08-17'].copy()
        d['volume'] = np.nan
        return d
    return pd.read_pickle(f'data/C_{name}.pkl')


def frame(name, tf, k=1.0):
    os.makedirs(CACHE, exist_ok=True)
    fn = f'{CACHE}/{name}_{tf}_k{k}.pkl'
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    df = load_1h(name)
    base, X = build(df, tf)
    inv = name.endswith('_INV')
    btc_name = ('BTC_BITSTAMP' if name.startswith('BTC_BITSTAMP') else 'BTCUSDT') + ('_INV' if inv else '')
    if name.replace('_INV', '') in ('BTCUSDT', 'BTC_BITSTAMP'):
        ctx = X[BTC_CTX]
    else:
        bb, bx = build(load_1h(btc_name), tf)
        step = pd.Timedelta(tf)
        ctx = join_by_close(base.index, step, bx[BTC_CTX], step, '')
    ctx = ctx.copy()
    ctx.columns = ['btc_' + c for c in ctx.columns]
    X = X.join(ctx)
    X['rel_mom_96'] = X['mom_96'] - X['btc_mom_96']
    X['rel_mom_384'] = X['mom_384'] - X['btc_mom_384']
    lab, fin = oracle_labels(base.close.values, k, return_final=True)
    out = pd.concat([base[['open', 'high', 'low', 'close']], X], axis=1)
    out['y'] = lab
    out['final'] = np.arange(len(out)) <= fin  # label cannot change with more data
    out['asset'] = name
    out['tf'] = tf
    out.to_pickle(fn)
    return out


def train_rows(fr, cutoff, k=1.0, embargo_bars=24):
    """rows usable for training at `cutoff`: labels recomputed on data up to the cutoff only, kept only while
    final (Viterbi merge point), then an extra embargo. No bar at/after the cutoff is ever touched."""
    step = pd.Timedelta(fr.tf.iloc[0])
    past = fr[fr.index + step <= cutoff]
    if len(past) < 3000:
        return past.iloc[:0]
    lab, fin = oracle_labels(past.close.values, k, return_final=True)
    keep = max(fin - embargo_bars, 0)
    tr = past.iloc[:keep].copy()
    tr['y'] = lab[:keep]
    return tr

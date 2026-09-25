"""
Market-breadth features: at each bar, how the whole crypto basket is trending (share of coins in an
up-state, mean normalised momentum). Built only from other coins' features at the SAME bar close, so it is
causal. Coins that are not listed yet are simply absent (nan-mean over what exists at that time).
Inverted (training-only) copies get breadth from the inverted basket, so the symmetry is preserved.
"""
import numpy as np
import pandas as pd
from .dataset import frame, COINS

B_SRC = {'mom_96': 'mean', 'mom_384': 'mean', 'dc_1.0': 'up', 'dc_2.0': 'up', 'ema_dist_200': 'up',
         'tstat_96': 'mean', 'h24_dc_1.0': 'up', 'h4_mom_96': 'mean'}


def breadth_table(tf, inv=False):
    suf = '_INV' if inv else ''
    cols = {}
    for a in COINS:
        fr = frame(a + suf, tf)
        for f, how in B_SRC.items():
            v = fr[f]
            cols.setdefault(f, []).append((v > 0).astype(float).where(v.notna()) if how == 'up' else v.clip(-5, 5))
    out = {}
    for f, lst in cols.items():
        m = pd.concat(lst, axis=1)
        out[f'br_{f}'] = m.mean(axis=1, skipna=True)
    out['br_n'] = pd.concat(cols['mom_96'], axis=1).notna().sum(axis=1)
    return pd.DataFrame(out)


_CACHE = {}


def with_breadth(fr):
    name, tf = fr.asset.iloc[0], fr.tf.iloc[0]
    inv = name.endswith('_INV')
    if name.startswith('BTC_BITSTAMP'):
        # before 2017-08 only BTC exists: breadth = BTC itself
        b = pd.DataFrame({f'br_{f}': ((fr[f] > 0).astype(float).where(fr[f].notna()) if h == 'up' else fr[f].clip(-5, 5))
                          for f, h in B_SRC.items()}, index=fr.index)
        b['br_n'] = 1.0
    else:
        key = (tf, inv)
        if key not in _CACHE:
            _CACHE[key] = breadth_table(tf, inv)
        b = _CACHE[key].reindex(fr.index)
    return pd.concat([fr, b.astype(np.float32)], axis=1)

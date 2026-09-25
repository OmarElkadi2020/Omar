"""Proposed label v2 (for review): pivot-weighted, asymmetric, with forward volume-flow confirmation."""
import numpy as np
import pandas as pd
from .labels import oracle_labels, segments, trailing_sigma

FLOOR, TAU_FRAC, TAU_MIN = 0.35, 0.15, 3      # pivot emphasis
M_FWD = 10                                     # look-ahead bars for volume flow
REBOUND_IN_DOWN = 0.5                          # counter-moves inside a downtrend count half


def label_v2(o, h, l, c, v, k_major=1.0, k_minor=0.3):
    c = np.asarray(c, float)
    lc = np.log(c)
    n = len(c)
    major = oracle_labels(c, k_major)
    minor = oracle_labels(c, k_minor)
    # 1) pivot emphasis: 1 at the reversal candle, decays to FLOOR along the trend
    piv = np.zeros(n)
    st, en, d = segments(major)
    move = np.zeros(n)
    for a, b, dd in zip(st, en, d):
        tau = max(TAU_MIN, TAU_FRAC * (b - a))
        t = np.arange(b - a)
        piv[a:b] = FLOOR + (1 - FLOOR) * np.exp(-t / tau)
        move[a:b] = abs(lc[min(b, n - 1)] - lc[a]) + 1e-9
    # 2) forward volume flow (look-ahead): share of volume on up-bars minus down-bars over the next M bars
    r = np.r_[0.0, np.diff(lc)]
    vol = np.nan_to_num(np.asarray(v, float))
    up = pd.Series(np.where(r > 0, vol, 0.0)).rolling(M_FWD).sum().shift(-M_FWD).values
    dn = pd.Series(np.where(r < 0, vol, 0.0)).rolling(M_FWD).sum().shift(-M_FWD).values
    flow = np.nan_to_num((up - dn) / np.maximum(up + dn, 1e-9))
    vconf = 0.75 + 0.25 * major * flow              # 0.5 (flow against trend) .. 1.0 (flow with trend)
    base = major * piv * vconf
    sig = trailing_sigma(c)
    sig = np.where(np.isnan(sig), np.nanmedian(sig[: max(200, n // 10)]), sig)
    flip_size = 2 * k_major * sig * np.sqrt(24.0)
    # 3) counter-moves (minor swing against the major trend): negative contribution by relative size,
    #    rebounds inside a DOWN trend count only half
    cm = np.zeros(n)
    s2, e2, d2 = segments(minor)
    for a, b, dd in zip(s2, e2, d2):
        maj = major[a]
        if dd == maj:
            continue
        size = abs(lc[min(b, n - 1)] - lc[a]) / flip_size[a]   # 1 = as big as a move that would flip the trend
        g = REBOUND_IN_DOWN if maj == -1 else 1.0
        cm[a:b] = dd * min(size, 1.0) * g
    y = np.clip(base + cm, -1, 1)
    return pd.DataFrame(dict(y=y, major=major, minor=minor, pivot=piv, flow=flow, counter=cm))

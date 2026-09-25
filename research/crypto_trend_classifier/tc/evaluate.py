"""Metrics of a causal trend state (+1/-1 per bar) against oracle labels, plus the classic baselines."""
import numpy as np
import pandas as pd
import numba
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from .features import _atr, _supertrend_dir, _dc_state
from .labels import ewm_mean, segments


# ------------------------------------------------------------------ smoothing of model probabilities
@numba.njit(cache=True)
def hysteresis(p, h):
    s = np.empty(len(p), np.int8)
    cur = 1 if p[0] >= 0.5 else -1
    for i in range(len(p)):
        if np.isnan(p[i]):
            s[i] = cur
            continue
        if cur == 1 and p[i] < 0.5 - h:
            cur = -1
        elif cur == -1 and p[i] > 0.5 + h:
            cur = 1
        s[i] = cur
    return s


def smooth_state(p, span, h):
    ps = ewm_mean(np.nan_to_num(p, nan=0.5), max(span, 1) / 2.0) if span > 1 else np.asarray(p, float)
    return hysteresis(ps, h)


# ------------------------------------------------------------------ baselines (all causal)
def ema_cross(fr, fast, slow):
    c = fr.close
    return np.where(c.ewm(span=fast, adjust=False).mean() > c.ewm(span=slow, adjust=False).mean(), 1, -1).astype(np.int8)


def price_ema(fr, n):
    c = fr.close
    return np.where(c > c.ewm(span=n, adjust=False).mean(), 1, -1).astype(np.int8)


def supertrend(fr, period, factor):
    a = _atr(fr, period).values
    d = _supertrend_dir(fr.high.values, fr.low.values, fr.close.values, a, factor)
    s = np.sign(d)
    s[s == 0] = 1
    return s.astype(np.int8)


def dc_online(fr, k):
    lc = np.log(fr.close.values)
    r = np.diff(lc, prepend=lc[0])
    sig = np.sqrt(ewm_mean(r * r, 720))
    th = k * sig * np.sqrt(24)
    th[:500] = np.nan
    st, _, _ = _dc_state(lc, th)
    s = pd.Series(st).replace(0, np.nan).ffill().fillna(1).values
    return s.astype(np.int8)


BASELINE_GRID = {
    'EMA cross': [dict(fast=f, slow=s) for f in (5, 10, 20, 30, 50) for s in (50, 100, 200, 300, 400) if f < s],
    'Price vs EMA': [dict(n=n) for n in (20, 50, 100, 200, 300, 400, 600)],
    'SuperTrend': [dict(period=p, factor=f) for p in (7, 10, 14, 24, 48) for f in (1.5, 2, 3, 4, 5)],
    'Online directional-change': [dict(k=k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)],
}
BASELINE_FN = {'EMA cross': lambda fr, p: ema_cross(fr, **p), 'Price vs EMA': lambda fr, p: price_ema(fr, **p),
               'SuperTrend': lambda fr, p: supertrend(fr, **p), 'Online directional-change': lambda fr, p: dc_online(fr, **p)}


# ------------------------------------------------------------------ metrics
def metrics(s, y, close, fee=0.0005, bars_per_year=8760):
    s = np.asarray(s, np.int8)
    y = np.asarray(y, np.int8)
    c = np.asarray(close, float)
    rn = np.r_[np.diff(np.log(c)), 0.0]
    out = dict(n=len(s), acc=(s == y).mean(), bal_acc=balanced_accuracy_score(y, s), mcc=matthews_corrcoef(y, s))
    o = np.sum(y * rn)
    out['capture'] = np.sum(s * rn) / o if o else np.nan
    fs = np.sum(s[1:] != s[:-1])
    fy = np.sum(y[1:] != y[:-1])
    out['flips_per_oracle_flip'] = fs / max(fy, 1)
    fl = np.flatnonzero(s[1:] != s[:-1]) + 1
    out['flip_precision'] = np.mean(y[fl] == s[fl]) if len(fl) else np.nan  # new direction == true trend at that moment
    st, en, d = segments(y)
    lagf, miss, give = [], [], []
    for a, b, dd in zip(st, en, d):
        if b - a < 3 or b >= len(s):
            continue
        hit = np.flatnonzero(s[a:b] == dd)
        mv = np.log(c[b] / c[a])
        if len(hit) == 0:
            lagf.append(1.0)
            miss.append(1.0)
            continue
        lagf.append(hit[0] / (b - a))
        miss.append(np.log(c[a + hit[0]] / c[a]) / mv if mv != 0 else 0)
        post = np.flatnonzero(s[b:] != dd)
        if len(post):
            give.append(abs(np.log(c[b + post[0]] / c[b])) / abs(mv) if mv != 0 else 0)
    out['entry_lag_frac'] = np.median(lagf) if lagf else np.nan
    out['missed_move_frac'] = np.median(miss) if miss else np.nan
    out['exit_giveback_frac'] = np.median(give) if give else np.nan
    pos = s.astype(float)
    pnl = pos * rn - fee * np.abs(np.diff(pos, prepend=pos[0]))
    out['ls_sharpe'] = pnl.mean() / pnl.std() * np.sqrt(bars_per_year) if pnl.std() > 0 else np.nan
    lf = np.maximum(pos, 0)
    pnl2 = lf * rn - fee * np.abs(np.diff(lf, prepend=lf[0]))
    out['lf_sharpe'] = pnl2.mean() / pnl2.std() * np.sqrt(bars_per_year) if pnl2.std() > 0 else np.nan
    bh = rn
    out['bh_sharpe'] = bh.mean() / bh.std() * np.sqrt(bars_per_year)
    return out


def delayed_oracle_curve(y, lags=(1, 2, 4, 8, 12, 24, 48, 96, 192)):
    """accuracy of the PERFECT labeler if it learned the truth L bars late: the causal reference line."""
    y = np.asarray(y)
    return {L: np.mean(y[L:] == y[:-L]) for L in lags}

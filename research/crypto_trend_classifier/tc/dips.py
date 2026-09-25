"""Stage 4 - the buy-the-dip permission filter (see PREREG_stage4.md)."""
import numpy as np
import pandas as pd
import numba
from .features import _atr
from .labels import oracle_labels, segments, ewm_mean
from .evaluate import BASELINE_FN

COOLDOWN, TP, SL, HOLD, COST = 12, 3.0, 3.0, 72, 0.001


def dip_events(fr):
    c = fr.close
    mid = c.rolling(20).mean()
    lo = mid - 2 * c.rolling(20).std()
    cross = (c < lo) & (c.shift(1) >= lo.shift(1))
    idx, last = [], -10 ** 9
    for i in np.flatnonzero(cross.values):
        if i - last >= COOLDOWN:
            idx.append(i)
            last = i
    return np.array(idx, int)


@numba.njit(cache=True)
def _trades(h, l, c, atr, ev):
    R = np.full(len(ev), np.nan)
    for j in range(len(ev)):
        i = ev[j]
        a = atr[i]
        if not (a > 0) or i + 1 >= len(c):
            continue
        e = c[i]
        tp, sl = e + TP * a, e - SL * a
        px = np.nan
        end = min(i + HOLD, len(c) - 1)
        for t in range(i + 1, end + 1):
            if l[t] <= sl:
                px = sl
                break
            if h[t] >= tp:
                px = tp
                break
        if np.isnan(px):
            px = c[end]
        R[j] = ((px - e) - COST * e) / (SL * a)
    return R


def dip_trades(fr):
    ev = dip_events(fr)
    atr = _atr(fr, 14).values
    R = _trades(fr.high.values, fr.low.values, fr.close.values, atr, ev)
    ok = ~np.isnan(R)
    return ev[ok], R[ok]


@numba.njit(cache=True)
def asym_state(p, a, b):
    s = np.zeros(len(p), np.int8)
    cur = 0
    for i in range(len(p)):
        if np.isnan(p[i]):
            s[i] = cur
            continue
        if cur == 0 and p[i] > a:
            cur = 1
        elif cur == 1 and p[i] < b:
            cur = 0
        s[i] = cur
    return s


def model_on(p, span, a, b):
    ps = ewm_mean(np.nan_to_num(p, nan=0.5), span / 2.0) if span > 1 else np.asarray(p, float)
    return asym_state(ps, a, b)


def price_ema_on(fr, n):
    c = fr.close
    return (c > c.ewm(span=n, adjust=False).mean()).values.astype(np.int8)


def baseline_on(fr, fam, prm):
    if fam == 'Price vs EMA':
        return price_ema_on(fr, prm['n'])
    return (np.asarray(BASELINE_FN[fam](fr, prm)) == 1).astype(np.int8)


GRIDS = {'Price vs EMA': [dict(n=n) for n in (20, 50, 100, 150, 200, 300)],
         'EMA cross': [dict(fast=f, slow=s) for f in (5, 10, 20, 30, 50) for s in (50, 100, 200, 300, 400) if f < s],
         'SuperTrend': [dict(period=p, factor=f) for p in (7, 10, 14, 24, 48) for f in (1.5, 2, 3, 4, 5)],
         'Online directional-change': [dict(k=k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]}
MODEL_GRID = [(sp, a, b) for sp in (1, 4, 12, 24, 48) for a in np.round(np.arange(.5, .801, .05), 2)
              for b in np.round(np.arange(.2, .501, .05), 2) if b <= a]


def tstat(R):
    R = np.asarray(R)
    if len(R) < 5 or R.std() == 0:
        return np.nan
    return R.mean() / R.std() * np.sqrt(len(R))


def diagnostics(on, close, lab2):
    """lab2 = ideal labels at k=2 on the same bars."""
    st, en, d = segments(lab2)
    fo, crash, n_up = 0, [], 0
    for a, b, dd in zip(st, en, d):
        if b - a < 3:
            continue
        seg = on[a:b]
        if dd == 1:
            n_up += 1
            fo += int(np.sum((seg[1:] == 0) & (seg[:-1] == 1)))
        else:
            fall = np.log(close[min(b, len(close) - 1)] / close[a])
            if fall >= 0:
                continue
            off = np.flatnonzero(seg == 0)
            crash.append(1.0 if len(off) == 0 else np.log(close[a + off[0]] / close[a]) / fall)
    return dict(false_offs_per_uptrend=fo / max(n_up, 1), crash_done_before_off=np.median(crash) if crash else np.nan,
                on_in_bear=float(on[lab2 == -1].mean()) if (lab2 == -1).any() else np.nan,
                on_in_bull=float(on[lab2 == 1].mean()) if (lab2 == 1).any() else np.nan)

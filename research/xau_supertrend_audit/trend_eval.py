"""
Evaluate the XAUSTBreakFree regime (4h SuperTrend + EMA filter [+ ADX gate]) as a TREND CLASSIFIER
against ideal look-ahead labels.

Ground truth ("oracle") labelers, both use the future on purpose:
  1. Oracle labeling (Kovacevic et al., IEEE Access 2023): dynamic programming that picks the
     position path {-1, 0, +1} maximising cumulative log return minus a cost `c` per unit of
     position change. The cost is the knob that ignores small pullbacks: a move against the trend
     smaller than ~2c is not worth flipping for. 3 classes -> matches the regime's {-1, 0, +1}.
  2. ZigZag / Directional-Change / Continuous-Trend-Labeling (Wu et al. 2020, Glattfelder et al.):
     a peak is confirmed only when price later falls `theta` from it before making a new high
     (and vice-versa). Every bar between two pivots is labelled with the segment's direction.

Trend scale is expressed in units of the asset's typical daily move (sigma_d = median yearly
std of daily log returns), so gold, BTC and EURUSD are compared on the same footing.
"""
import numpy as np
import pandas as pd
from xau_bt import supertrend, ema, adx, session_mask


# ------------------------------------------------------------------ oracle labelers
def oracle_dp(close, cost, allow_flat=True):
    """label[t] = position to hold over (t, t+1]. cost = log-return cost per unit position change."""
    r = np.diff(np.log(close))  # r[t] = return over (t, t+1]
    S = np.array([-1, 0, 1]) if allow_flat else np.array([-1, 1])
    K, N = len(S), len(r)
    V = np.zeros(K)
    back = np.zeros((N, K), dtype=np.int8)
    trans = cost * np.abs(S[:, None] - S[None, :])  # [prev, new]
    for t in range(N):
        tot = V[:, None] - trans           # prev -> new
        bi = tot.argmax(0)
        back[t] = bi
        V = tot[bi, np.arange(K)] + S * r[t]
    lab = np.zeros(N + 1, dtype=np.int8)
    k = int(V.argmax())
    for t in range(N - 1, -1, -1):
        lab[t] = S[k]
        k = back[t, k]
    lab[N] = lab[N - 1]
    return lab


def zigzag(close, theta):
    """pivots (indices) and per-bar direction labels; theta = fractional reversal threshold."""
    n = len(close)
    piv = [0]
    d = 0
    ext_i = 0
    hi_i = lo_i = 0
    for i in range(1, n):
        p = close[i]
        if d == 0:
            if close[i] > close[hi_i]:
                hi_i = i
            if close[i] < close[lo_i]:
                lo_i = i
            if p >= close[lo_i] * (1 + theta):
                d, piv[0], ext_i = 1, lo_i, i
            elif p <= close[hi_i] * (1 - theta):
                d, piv[0], ext_i = -1, hi_i, i
        elif d == 1:
            if p > close[ext_i]:
                ext_i = i
            elif p <= close[ext_i] * (1 - theta):
                piv.append(ext_i); d = -1; ext_i = i
        else:
            if p < close[ext_i]:
                ext_i = i
            elif p >= close[ext_i] * (1 + theta):
                piv.append(ext_i); d = 1; ext_i = i
    piv.append(ext_i if d != 0 else n - 1)
    piv = np.array(sorted(set(piv)))
    lab = np.zeros(n, dtype=np.int8)
    for a, b in zip(piv[:-1], piv[1:]):
        lab[a:b] = 1 if close[b] > close[a] else -1
    lab[piv[-1]:] = lab[piv[-1] - 1] if piv[-1] > 0 else 0
    return piv, lab


# ------------------------------------------------------------------ classifiers (causal)
def resample(df, rule):
    return df.resample(rule, origin='epoch').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()


def regime_classifier(grid, anchor_rule, session, st_period=24, st_factor=4.0, ema_n=10,
                      l_adx=10, s_adx=15, exclude_last=True):
    """Strategy regime evaluated at each grid close. Returns (regime, regime_with_adx_gate)."""
    anc = resample(grid, anchor_rule)
    anc = anc[session_mask(anc.index, session)]
    H, L, C = anc.high.values, anc.low.values, anc.close.values
    st = supertrend(H, L, C, st_period, st_factor)
    em = ema(C, ema_n) if ema_n > 0 else None
    ax = adx(H, L, C, 14)
    step = pd.Timedelta(anchor_rule)
    if exclude_last:
        # exactly like get_candles(tf)[:-1]: drop the anchor candle that contains the current bar
        pos = np.searchsorted(anc.index.values, grid.index.floor(step).values, side='left') - 1
    else:
        # anchor == grid: last completed candle is the current bar itself
        pos = np.searchsorted(anc.index.values, grid.index.values, side='right') - 1
    c = grid.close.values
    ok = pos >= 0
    p = np.where(ok, pos, 0)
    line = np.where(ok, st[p], np.nan)
    reg = np.where(np.isnan(line) | (line <= 0), 0, np.where(c > line, 1, -1))
    if em is not None:
        e = np.where(ok, em[p], np.nan)
        reg = np.where(np.isnan(e), 0, reg)
        reg = np.where((reg == 1) & (c < e), 0, reg)
        reg = np.where((reg == -1) & (c > e), 0, reg)
    a = np.where(ok, ax[p], np.nan)
    gated = reg.copy()
    gated[(reg == 1) & ~(a >= l_adx)] = 0
    gated[(reg == -1) & ~(a >= s_adx)] = 0
    return reg.astype(np.int8), gated.astype(np.int8)


def ema_baseline(grid, n):
    c = grid.close.values
    e = ema(c, n)
    return np.where(np.isnan(e), 0, np.where(c > e, 1, -1)).astype(np.int8)


# ------------------------------------------------------------------ metrics
def mcc_multiclass(y, p, classes=(-1, 0, 1)):
    C = np.array([[np.sum((y == a) & (p == b)) for b in classes] for a in classes], float)
    t, s = C.sum(1), C.sum(0)
    c, n = np.trace(C), C.sum()
    den = np.sqrt((n * n - (s * s).sum()) * (n * n - (t * t).sum()))
    return (c * n - (s * t).sum()) / den if den > 0 else 0.0


def capture(pred, lab, rnext):
    o = np.sum(lab * rnext)
    return np.sum(pred * rnext) / o if o != 0 else np.nan


def basic_scores(pred, lab, rnext):
    nz = pred != 0
    out = dict(
        coverage=nz.mean(),
        hit_when_active=(pred[nz] == lab[nz]).mean() if nz.any() else np.nan,   # says up/down and it's right
        wrong_way=((pred != 0) & (pred == -lab)).mean(),                          # share of ALL time spent against the oracle trend
        wrong_way_when_active=(pred[nz] == -lab[nz]).mean() if nz.any() else np.nan,
        up_precision=(lab[pred == 1] == 1).mean() if (pred == 1).any() else np.nan,
        up_recall=(pred[lab == 1] == 1).mean() if (lab == 1).any() else np.nan,
        dn_precision=(lab[pred == -1] == -1).mean() if (pred == -1).any() else np.nan,
        dn_recall=(pred[lab == -1] == -1).mean() if (lab == -1).any() else np.nan,
        mcc=mcc_multiclass(lab, pred),
        capture=capture(pred, lab, rnext),
        flips_per_1000=np.sum(pred[1:] != pred[:-1]) / len(pred) * 1000,
    )
    return out


def segment_scores(pred, close, piv):
    """per ZigZag trend: entry lag, share of move missed, exit giveback, false end signals, wrong-way flips."""
    rows = []
    for k in range(len(piv) - 1):
        a, b = piv[k], piv[k + 1]
        if b - a < 2:
            continue
        d = 1 if close[b] > close[a] else -1
        move = np.log(close[b] / close[a])
        seg = pred[a:b]
        hit = np.where(seg == d)[0]
        detected = len(hit) > 0
        row = dict(dir=d, bars=b - a, move=abs(move), detected=detected)
        if detected:
            t0 = a + hit[0]
            row['entry_lag_bars'] = hit[0]
            row['missed_frac'] = np.log(close[t0] / close[a]) / move  # share of the trend gone before detection
            after = pred[t0:b]
            row['false_exits'] = int(np.sum((after[1:] != d) & (after[:-1] == d)))
            row['wrong_way_flips'] = int(np.sum((after[1:] == -d) & (after[:-1] != -d)))
            # exit: first bar at/after the peak b where classifier no longer says d
            post = pred[b:]
            ex = np.where(post != d)[0]
            if len(ex):
                te = b + ex[0]
                row['exit_lag_bars'] = ex[0]
                row['giveback_frac'] = abs(np.log(close[te] / close[b])) / abs(move)  # of the trend's own move
                row['captured_frac'] = d * np.log(close[te] / close[t0]) / abs(move)
        rows.append(row)
    return pd.DataFrame(rows)


def circular_null(pred, lab, rnext, n=200, seed=0):
    rng = np.random.default_rng(seed)
    N = len(pred)
    hits, caps, mccs, ww = [], [], [], []
    for _ in range(n):
        s = rng.integers(N // 10, N - N // 10)
        q = np.roll(pred, s)
        nz = q != 0
        hits.append((q[nz] == lab[nz]).mean())
        caps.append(capture(q, lab, rnext))
        mccs.append(mcc_multiclass(lab, q))
        ww.append(((q != 0) & (q == -lab)).mean())
    return np.array(hits), np.array(caps), np.array(mccs), np.array(ww)

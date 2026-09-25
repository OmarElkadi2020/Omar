"""
Oracle trend labels (look-ahead ON PURPOSE - they are the target, never a feature).

Oracle labeling (Kovacevic et al., IEEE Access 2023): the position path s_t in {-1,+1} that maximises
    sum_t s_t * r_{t+1}  -  sum_t cost_t * |s_t - s_{t-1}|
solved exactly with dynamic programming (Viterbi). The switching cost is what makes it ignore pullbacks.

cost_t = k * sigma_bar(t) * sqrt(24): a volatility-scaled cost expressed in bar units, so the same k gives
trends of the same length *in bars* on 1h and 4h candles and on calm or wild coins. sigma_bar(t) is a
trailing (causal) EWM volatility, so the label definition itself never uses a future volatility estimate.

Finality (anti-leakage for training): when the series is truncated at a cutoff T, labels of the last
bars are not final - more data could change them. With Viterbi, every survivor path at T shares the same
prefix up to the point where they merge; that prefix is identical for ANY future continuation, so it is
final. `oracle_labels(..., return_final=True)` returns the index of the last final label; training only
ever uses labels at or before it.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _dp2(r, cost):
    N = len(r)
    back = np.zeros((N, 2), np.int8)  # state 0 = short (-1), 1 = long (+1)
    v0, v1 = 0.0, 0.0
    for t in range(N):
        c2 = 2.0 * cost[t]
        # new short
        a, b = v0, v1 - c2
        if a >= b:
            n0, back[t, 0] = a, 0
        else:
            n0, back[t, 0] = b, 1
        # new long
        a, b = v1, v0 - c2
        if a >= b:
            n1, back[t, 1] = a, 1
        else:
            n1, back[t, 1] = b, 0
        v0, v1 = n0 - r[t], n1 + r[t]
    lab = np.zeros(N, np.int8)
    k = 1 if v1 >= v0 else 0
    # backtrack best path, and simultaneously the other survivor to find the merge point
    ka, kb = 1, 0
    merge = -1
    for t in range(N - 1, -1, -1):
        lab[t] = 1 if k == 1 else -1
        if merge < 0 and ka == kb:
            merge = t
        k = back[t, k]
        ka = back[t, ka]
        kb = back[t, kb]
    return lab, merge


def trailing_sigma(close, halflife=168):
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    s = np.sqrt(ewm_mean(lr * lr, halflife))
    s[:24] = np.nan
    return s


@numba.njit(cache=True)
def _ewm(x, alpha):
    out = np.empty_like(x)
    m = x[0]
    for i in range(len(x)):
        m = alpha * x[i] + (1 - alpha) * m
        out[i] = m
    return out


def ewm_mean(x, halflife):
    return _ewm(np.asarray(x, float), 1 - np.exp(np.log(0.5) / halflife))


def oracle_labels(close, k, return_final=False):
    """label[t] = trend over (t, t+1]; last bar copies previous. final_idx = last index whose label is final."""
    close = np.asarray(close, float)
    sig = trailing_sigma(close)
    sig = np.where(np.isnan(sig), np.nanmedian(sig[: max(200, len(sig) // 10)]), sig)
    cost = k * sig * np.sqrt(24.0)
    r = np.diff(np.log(close))
    lab, merge = _dp2(r, cost[:-1])
    lab = np.r_[lab, lab[-1]]
    if return_final:
        return lab, max(merge, 0)
    return lab


def segments(lab):
    """(start, end_exclusive, direction) runs of the label series."""
    ch = np.flatnonzero(np.diff(lab)) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(lab)]
    return st, en, lab[st]

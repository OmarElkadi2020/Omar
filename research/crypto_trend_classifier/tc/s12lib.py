"""Stage 12 building blocks: new causal features, new labels, signal conversion and lag metrics (numba)."""
import numpy as np
import pandas as pd
import numba
from .labels import trailing_sigma, ewm_mean


# ------------------------------------------------------------------ features (all causal)
@numba.njit(cache=True)
def _cusum(z, kappa, h):
    n = len(z)
    sp, sn = np.zeros(n), np.zeros(n)
    agep, agen = np.full(n, 500.0), np.full(n, 500.0)
    a, b, lp, ln = 0.0, 0.0, -100000, -100000
    for t in range(n):
        x = z[t]
        if not np.isnan(x):
            a = max(0.0, a + x - kappa)
            b = max(0.0, b - x - kappa)
        if a > h:
            lp = t
        if b > h:
            ln = t
        sp[t], sn[t] = a, b
        agep[t], agen[t] = min(t - lp, 500), min(t - ln, 500)
    return sp, sn, agep, agen


@numba.njit(cache=True)
def _bocpd(z, hazard, tau2, rmax):
    n = len(z)
    p = np.zeros(rmax + 1)
    S = np.zeros(rmax + 1)
    N = np.zeros(rmax + 1)
    p[0] = 1.0
    o5, o20, oer = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        x = z[t]
        if np.isnan(x):
            continue
        pred = np.zeros(rmax + 1)
        for r in range(rmax + 1):
            prec = N[r] + 1.0 / tau2
            mu = S[r] / prec
            var = 1.0 + 1.0 / prec
            pred[r] = np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        q = np.zeros(rmax + 1)
        S2 = np.zeros(rmax + 1)
        N2 = np.zeros(rmax + 1)
        cp = 0.0
        for r in range(rmax + 1):
            w = p[r] * pred[r]
            cp += w * hazard
            rr = min(r + 1, rmax)
            if rr == rmax and r == rmax:     # truncation bucket keeps accumulating
                q[rr] += w * (1 - hazard)
                S2[rr] = S[r] + x
                N2[rr] = N[r] + 1
            else:
                q[rr] += w * (1 - hazard)
                S2[rr] = S[r] + x
                N2[rr] = N[r] + 1
        q[0] = cp
        S2[0] = x
        N2[0] = 1.0
        tot = q.sum()
        if tot <= 0 or np.isnan(tot):
            q[:] = 0.0
            q[0] = 1.0
            tot = 1.0
        p = q / tot
        S, N = S2, N2
        o5[t] = p[:5].sum()
        o20[t] = p[:20].sum()
        e = 0.0
        for r in range(rmax + 1):
            e += r * p[r]
        oer[t] = e
    return o5, o20, oer


def new_features(o, h, l, c, v):
    """o,h,l,c,v: pandas Series on the bar index."""
    lc = np.log(c)
    r = lc.diff()
    sig = np.sqrt(pd.Series(ewm_mean(np.nan_to_num(r.values ** 2), 20), c.index))
    z = (r / sig.shift(1)).clip(-8, 8)
    F = pd.DataFrame(index=c.index)
    for k in (1, 5, 21, 63, 126, 252):
        F[f'nret_{k}'] = (lc - lc.shift(k)) / (sig * np.sqrt(k))
    for s, L in ((8, 24), (16, 48), (32, 96)):
        q = (c.ewm(span=s, adjust=False).mean() - c.ewm(span=L, adjust=False).mean()) / c.rolling(63).std()
        F[f'macdn_{s}_{L}'] = q / q.rolling(252).std()
    zz = z.values.astype(float)
    for kap in (0.25, 0.5, 1.0):
        sp, sn, ap, an = _cusum(zz, kap, 4.0)
        F[f'cusum_p_{kap}'], F[f'cusum_n_{kap}'] = sp, sn
        F[f'cusum_age_p_{kap}'], F[f'cusum_age_n_{kap}'] = ap, an
        F[f'cusum_d_{kap}'] = sp - sn
    b5, b20, ber = _bocpd(zz, 1.0 / 100, 0.1, 300)
    F['bocpd_p5'], F['bocpd_p20'], F['bocpd_erl'] = b5, b20, ber
    if v is not None and np.nansum(v.values) > 0:
        obv = (np.sign(r).fillna(0) * v.fillna(0)).cumsum()
        rng = (h - l).replace(0, np.nan)
        clv = (((c - l) - (h - c)) / rng).fillna(0)
        adl = (clv * v.fillna(0)).cumsum()
        for n in (20, 60):
            vs = v.fillna(0).rolling(n).sum().replace(0, np.nan)
            F[f'obv_slope_{n}'] = (obv - obv.shift(n)) / vs
            F[f'ad_slope_{n}'] = (adl - adl.shift(n)) / vs
    else:
        for n in (20, 60):
            F[f'obv_slope_{n}'] = np.nan
            F[f'ad_slope_{n}'] = np.nan
    return F.astype(np.float32)


# ------------------------------------------------------------------ labels
@numba.njit(cache=True)
def _dp3(r, cost):
    """ternary oracle: states 0=down(-1),1=neutral(0),2=up(+1); up<->down forbidden; neutral earns 0."""
    N = len(r)
    back = np.zeros((N, 3), np.int8)
    v = np.zeros(3)
    pos = np.array([-1.0, 0.0, 1.0])
    NEG = -1e18
    for t in range(N):
        nv = np.empty(3)
        for s in range(3):
            best, arg = NEG, 0
            for q in range(3):
                if abs(s - q) == 2:
                    continue
                val = v[q] - cost[t] * abs(pos[s] - pos[q])
                if val > best:
                    best, arg = val, q
            nv[s] = best + pos[s] * r[t]
            back[t, s] = arg
        v = nv
    lab = np.zeros(N, np.int8)
    k = int(np.argmax(v))
    ka, kb, kc = 0, 1, 2
    merge = -1
    for t in range(N - 1, -1, -1):
        lab[t] = k - 1
        if merge < 0 and ka == kb and kb == kc:
            merge = t
        k = back[t, k]
        ka, kb, kc = back[t, ka], back[t, kb], back[t, kc]
    return lab, merge


def ternary_labels(close, k=1.0):
    close = np.asarray(close, float)
    sig = trailing_sigma(close)
    sig = np.where(np.isnan(sig), np.nanmedian(sig[: max(200, len(sig) // 10)]), sig)
    cost = k * sig * np.sqrt(24.0)
    lab, merge = _dp3(np.diff(np.log(close)), cost[:-1])
    return np.r_[lab, lab[-1]], max(merge, 0)


@numba.njit(cache=True)
def trend_scan(x, lmin, lmax):
    """forward trend-scanning: t-value of the OLS slope of x[t..t+L], L in [lmin,lmax], max |t|."""
    n = len(x)
    tv = np.full(n, np.nan)
    for t in range(n - lmin):
        best = 0.0
        sx = sy = sxx = sxy = syy = 0.0
        for j in range(0, min(lmax, n - 1 - t) + 1):
            y = x[t + j]
            sx += j
            sy += y
            sxx += j * j
            sxy += j * y
            syy += y * y
            m = j + 1
            if j >= lmin:
                vx = sxx - sx * sx / m
                b = (sxy - sx * sy / m) / vx
                a = (sy - b * sx) / m
                sse = syy - a * sy - b * sxy
                if sse <= 0:
                    continue
                se = np.sqrt(sse / (m - 2) / vx)
                tt = b / se
                if abs(tt) > abs(best):
                    best = tt
        tv[t] = best
    return tv


def remaining_value(close, lab, fin):
    """log move left until the end of the current k=1 segment / sigma_21; nan where the segment end is not final."""
    c = np.asarray(close, float)
    lc = np.log(c)
    sig = np.sqrt(ewm_mean(np.nan_to_num(np.r_[0, np.diff(lc)] ** 2), 20)) * np.sqrt(21)
    out = np.full(len(c), np.nan)
    ch = np.flatnonzero(np.diff(lab)) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(lab)]
    for a, b in zip(st, en):
        if b > fin:
            break
        e = min(b, len(c) - 1)
        out[a:b] = (lc[e] - lc[a:b]) / np.maximum(sig[a:b], 1e-6)
    return out


# ------------------------------------------------------------------ signal conversion
@numba.njit(cache=True)
def conv_h(s, th_in, th_out):
    """3-state hysteresis on a score in [-1,1]. th_out = -th_in gives a binary switch."""
    n = len(s)
    st = np.zeros(n, np.int8)
    cur = 0
    for i in range(n):
        x = s[i]
        if np.isnan(x):
            st[i] = cur
            continue
        if cur == 1:
            if x < -th_in:
                cur = -1
            elif x < th_out:
                cur = 0
        elif cur == -1:
            if x > th_in:
                cur = 1
            elif x > -th_out:
                cur = 0
        else:
            if x > th_in:
                cur = 1
            elif x < -th_in:
                cur = -1
        st[i] = cur
    return st


@numba.njit(cache=True)
def conv_c(s, kappa, h):
    n = len(s)
    st = np.zeros(n, np.int8)
    gp = gn = 0.0
    cur = 0
    for i in range(n):
        x = s[i]
        if not np.isnan(x):
            gp = max(0.0, gp + x - kappa)
            gn = max(0.0, gn - x - kappa)
            if gp > h:
                cur, gp, gn = 1, 0.0, 0.0
            elif gn > h:
                cur, gp, gn = -1, 0.0, 0.0
        st[i] = cur
    return st


def ewm_score(s, span):
    return ewm_mean(np.nan_to_num(s, nan=0.0), span / 2.0) if span > 1 else np.asarray(s, float)


# ------------------------------------------------------------------ lag metrics vs binary truth
@numba.njit(cache=True)
def lag_metrics(state, truth, lc):
    """returns flips/true-flips, median missed-move, median delay (bars), median give-back, n segments."""
    n = len(state)
    fs = 0
    ft = 0
    for i in range(1, n):
        if state[i] != state[i - 1]:
            fs += 1
        if truth[i] != truth[i - 1]:
            ft += 1
    miss = []
    dly = []
    give = []
    a = 0
    for i in range(1, n + 1):
        if i == n or truth[i] != truth[i - 1]:
            b = i
            d = truth[a]
            if b - a >= 5 and b < n:
                mv = lc[b] - lc[a]
                hit = -1
                for j in range(a, b):
                    if state[j] == d:
                        hit = j
                        break
                if hit < 0:
                    miss.append(1.0)
                    dly.append(float(b - a))
                else:
                    m = (lc[hit] - lc[a]) / mv if mv != 0 else 0.0
                    miss.append(min(max(m, 0.0), 1.0))
                    dly.append(float(hit - a))
                    post = -1
                    for j in range(b, n):
                        if state[j] != d:
                            post = j
                            break
                    if post > 0 and mv != 0:
                        give.append(abs(lc[post] - lc[b]) / abs(mv))
            a = i
    fr = fs / max(ft, 1)
    mm = np.median(np.array(miss)) if len(miss) else np.nan
    md = np.median(np.array(dly)) if len(dly) else np.nan
    mg = np.median(np.array(give)) if len(give) else np.nan
    return fr, mm, md, mg, len(miss)

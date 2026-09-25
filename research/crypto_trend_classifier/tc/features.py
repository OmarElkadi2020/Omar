"""
Causal, scale-free features. Every feature at bar t uses candles that are COMPLETE at the close of bar t:
  - rolling / ewm windows only look backwards (no centred windows, no full-sample normalisation)
  - higher-timeframe (HTF) candles are joined by their CLOSE time (merge_asof backward), so a 4h candle is
    only visible after it has closed
  - everything is normalised by trailing volatility, so the same model works on 1h and 4h bars and on
    calm or wild coins
`leak_test()` in run scripts re-computes features on truncated data and checks they are identical.
"""
import numpy as np
import pandas as pd
import numba
from .labels import ewm_mean

WIN = [6, 12, 24, 48, 96, 192, 384]


@numba.njit(cache=True)
def _supertrend_dir(h, l, c, atr, factor):
    n = len(c)
    d = np.zeros(n)
    fub = np.nan
    flb = np.nan
    dd = 1
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        hl2 = (h[i] + l[i]) / 2
        ub, lb = hl2 + factor * atr[i], hl2 - factor * atr[i]
        if np.isnan(fub):
            fub, flb = ub, lb
        else:
            fub = ub if (ub < fub or c[i - 1] > fub) else fub
            flb = lb if (lb > flb or c[i - 1] < flb) else flb
        if dd == 1 and c[i] < flb:
            dd = -1
        elif dd == -1 and c[i] > fub:
            dd = 1
        d[i] = (c[i] - (flb if dd == 1 else fub)) / atr[i]  # >0 in up-trend (above lower band), <0 in down-trend
    return d


@numba.njit(cache=True)
def _dc_state(logp, theta):
    """causal directional-change state: +1 after a rise of theta from the running low, -1 after a fall of
    theta from the running high. Returns state, overshoot (move since confirmation / theta), bars since."""
    n = len(logp)
    st = np.zeros(n)
    ov = np.zeros(n)
    age = np.zeros(n)
    mode = 0
    hi = logp[0]
    lo = logp[0]
    ext = logp[0]
    conf_p = logp[0]
    conf_i = 0
    for i in range(n):
        p = logp[i]
        th = theta[i]
        if np.isnan(th):
            hi = max(hi, p)
            lo = min(lo, p)
            continue
        if mode == 0:
            hi = max(hi, p)
            lo = min(lo, p)
            if p >= lo + th:
                mode, ext, conf_p, conf_i = 1, p, p, i
            elif p <= hi - th:
                mode, ext, conf_p, conf_i = -1, p, p, i
        elif mode == 1:
            if p > ext:
                ext = p
            elif p <= ext - th:
                mode, ext, conf_p, conf_i = -1, p, p, i
        else:
            if p < ext:
                ext = p
            elif p >= ext + th:
                mode, ext, conf_p, conf_i = 1, p, p, i
        st[i] = mode
        ov[i] = (p - conf_p) / th if mode != 0 else 0.0
        age[i] = i - conf_i
    return st, ov, age


@numba.njit(cache=True)
def _since_ext(x, n, want_max):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        best = i - n + 1
        for j in range(i - n + 1, i + 1):
            if (want_max and x[j] >= x[best]) or ((not want_max) and x[j] <= x[best]):
                best = j
        out[i] = (i - best) / n
    return out


def _atr(df, n):
    pc = df.close.shift(1)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    pdi = pd.Series(pdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    ndi = pd.Series(ndm, df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), (pdi - ndi) / (pdi + ndi).replace(0, np.nan)


def core_features(df):
    """df: OHLCV indexed by candle OPEN time, one timeframe. Returns features indexed the same way."""
    c = df.close
    lc = np.log(c)
    r = lc.diff()
    F = {}
    sig = pd.Series(np.sqrt(ewm_mean((r.fillna(0) ** 2).values, 96)), df.index)  # per-bar vol
    sig_slow = pd.Series(np.sqrt(ewm_mean((r.fillna(0) ** 2).values, 720)), df.index)
    F['vol_ratio'] = np.log(sig / sig_slow)
    F['vol_level'] = np.log(sig_slow / sig_slow.rolling(2000, min_periods=200).median())
    sig[:200] = np.nan
    atr = _atr(df, 14)
    for n in WIN:
        F[f'mom_{n}'] = (lc - lc.shift(n)) / (sig * np.sqrt(n))
        hi, lo = c.rolling(n).max(), c.rolling(n).min()
        F[f'don_pos_{n}'] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5
        F[f'dd_{n}'] = np.log(c / hi) / (sig * np.sqrt(n))
        F[f'ru_{n}'] = np.log(c / lo) / (sig * np.sqrt(n))
        F[f'since_hi_{n}'] = _since_ext(c.values, n, True)
        F[f'since_lo_{n}'] = _since_ext(c.values, n, False)
        # efficiency ratio (Kaufman): net move / path length
        F[f'er_{n}'] = (lc - lc.shift(n)) / r.abs().rolling(n).sum()
    for n in [10, 20, 50, 100, 200, 400]:
        e = c.ewm(span=n, adjust=False, min_periods=n).mean()
        F[f'ema_dist_{n}'] = (c - e) / atr
        F[f'ema_slope_{n}'] = (np.log(e) - np.log(e.shift(max(n // 4, 2)))) / (sig * np.sqrt(max(n // 4, 2)))
    for f, s in [(10, 40), (20, 100), (50, 200)]:
        F[f'ema_x_{f}_{s}'] = (c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()) / atr
    # rolling regression slope t-stat (backward trend scanning)
    t = np.arange(len(c), dtype=float)
    for n in [24, 96, 384]:
        x = pd.Series(t, df.index)
        cov = lc.rolling(n).cov(x)
        var = x.rolling(n).var()
        beta = cov / var
        resid_var = (lc.rolling(n).var() - beta * beta * var).clip(lower=1e-18)
        F[f'tstat_{n}'] = beta / np.sqrt(resid_var / (var * (n - 2)))
    for n, fac in [(10, 3.0), (24, 4.0), (48, 3.0)]:
        a = _atr(df, n).values
        F[f'st_{n}_{fac}'] = _supertrend_dir(df.high.values, df.low.values, c.values, a, fac)
    adx, di = _adx(df, 14)
    F['adx14'] = adx
    F['di14'] = di
    adx2, di2 = _adx(df, 56)
    F['adx56'] = adx2
    F['di56'] = di2
    for n in [14, 56]:
        d = c.diff()
        g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        ls = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        F[f'rsi_{n}'] = g / (g + ls) - 0.5
    # causal directional-change states at vol-scaled thresholds (the online cousin of the oracle)
    lp = lc.values
    for k in [0.5, 1.0, 2.0, 4.0]:
        th = (k * sig_slow * np.sqrt(24)).values.copy()
        th[:500] = np.nan
        th[~(th > 0)] = np.nan
        st, ov, age = _dc_state(lp, th)
        F[f'dc_{k}'] = st
        F[f'dc_ov_{k}'] = np.clip(ov, -10, 10)
        F[f'dc_age_{k}'] = np.log1p(age)
    if 'volume' in df:
        lv = np.log(df.volume.replace(0, np.nan)).ffill()
        F['vol_z'] = (lv - lv.rolling(720, min_periods=100).mean()) / lv.rolling(720, min_periods=100).std()
        sv = np.sign(r) * df.volume
        F['flow_48'] = sv.rolling(48).sum() / df.volume.rolling(48).sum()
        F['flow_192'] = sv.rolling(192).sum() / df.volume.rolling(192).sum()
    out = pd.DataFrame(F, index=df.index).replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


HTF_KEEP = ['mom_24', 'mom_96', 'mom_384', 'ema_dist_50', 'ema_dist_200', 'ema_slope_50', 'ema_slope_200',
            'don_pos_96', 'dd_96', 'ru_96', 'er_96', 'tstat_96', 'st_24_4.0', 'adx14', 'di14', 'dc_1.0', 'dc_2.0',
            'dc_ov_1.0', 'vol_ratio']


def resample(df, rule):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df:
        agg['volume'] = 'sum'
    return df.resample(rule, origin='epoch').agg(agg).dropna(subset=['close'])


def join_by_close(base_idx, base_step, feat, feat_step, prefix):
    """attach feat (indexed by open time, candle length feat_step) to base bars (open time, length base_step)
    using only feat candles whose close time <= base bar close time."""
    left = pd.DataFrame({'t': base_idx + base_step})
    right = feat.copy()
    right.columns = [prefix + c for c in right.columns]
    right['t'] = feat.index + feat_step
    m = pd.merge_asof(left, right.sort_values('t'), on='t', direction='backward')
    m.index = base_idx
    return m.drop(columns='t')


def build(df, tf, htf_mult=(4, 24)):
    """features for one asset at timeframe tf ('1h','2h','4h',...) from 1h OHLCV df."""
    step = pd.Timedelta(tf)
    base = df if tf == '1h' else resample(df, tf)
    X = core_features(base)
    for m in htf_mult:
        hs = step * m
        hdf = resample(df, hs)
        hf = core_features(hdf)[HTF_KEEP]
        X = X.join(join_by_close(base.index, step, hf, hs, f'h{m}_'))
    return base, X

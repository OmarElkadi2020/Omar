"""
Python port of Jesse's XAUSTBreakFree strategy (1h trading, 4h anchor) for independent testing.

Faithful to the original logic:
  - CME metals session filter (America/New_York): Sun 18-24, Mon-Thu 00-17 & 18-24, Fri 00-17
  - anchor = completed 4h candles (get_candles('4h')[:-1]) filtered to session
  - regime = 4h SuperTrend side + optional 4h EMA filter, ADX(4h) threshold per side
  - entry = 1h close breaks the prior `don_period` session bars' Donchian channel, market fill at close
  - size = 3% risk to ATR stop, capped at 4x margin; futures P&L, fees 0.02% per side
  - long: ATR stop + ATR trailing stop; short: ATR stop + ATR trailing + ATR take profit

Execution differences vs Jesse (documented, all conservative or neutral):
  - 1h bars instead of 1m sub-bars: if stop and TP are both touched in the same bar, stop is assumed first
  - gaps through a stop fill at the bar open (Jesse fills at the stop price)
"""
import numpy as np
import pandas as pd

def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t


DEFAULT_HP = dict(st_period=24, st_factor=4.0, ema4=10, don_period=85, atr_stop=3.0, trail_atr=6.5,
                  exit_on_flip=0, l_adx=10, s_adx=15, s_stop=4.0, s_trail=3.5, s_tp=5.0)

HP_SPACE = [  # name, min, max, step (None -> int step 1), type
    ('st_period', 7, 40, 1, int), ('st_factor', 1.5, 5.0, 0.5, float), ('ema4', 0, 100, 10, int),
    ('don_period', 10, 100, 5, int), ('atr_stop', 1.0, 5.0, 0.5, float), ('trail_atr', 1.5, 8.0, 0.5, float),
    ('exit_on_flip', 0, 1, 1, int), ('l_adx', 0, 40, 1, int), ('s_adx', 0, 40, 1, int),
    ('s_stop', 0.5, 4.0, 0.5, float), ('s_trail', 1.0, 6.0, 0.5, float), ('s_tp', 0.0, 6.0, 0.5, float),
]


# ---------------------------------------------------------------- indicators (TA-Lib conventions)
def _rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def true_range(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n=14):
    tr = true_range(h, l, c)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    out[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def ema(c, n):
    out = np.full(len(c), np.nan)
    if n <= 0 or len(c) < n:
        return out
    k = 2 / (n + 1)
    out[n - 1] = c[:n].mean()
    for i in range(n, len(c)):
        out[i] = c[i] * k + out[i - 1] * (1 - k)
    return out


def adx(h, l, c, n=14):
    up = np.r_[np.nan, h[1:] - h[:-1]]
    dn = np.r_[np.nan, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    N = len(c)
    out = np.full(N, np.nan)
    if N < 2 * n + 1:
        return out
    str_, sp, sn = tr[1:n + 1].sum(), pdm[1:n + 1].sum(), ndm[1:n + 1].sum()
    dx = np.full(N, np.nan)
    for i in range(n, N):
        if i > n:
            str_ = str_ - str_ / n + tr[i]
            sp = sp - sp / n + pdm[i]
            sn = sn - sn / n + ndm[i]
        pdi, ndi = 100 * sp / str_, 100 * sn / str_
        dx[i] = 0 if pdi + ndi == 0 else 100 * abs(pdi - ndi) / (pdi + ndi)
    out[2 * n - 1] = np.mean(dx[n:2 * n])
    for i in range(2 * n, N):
        out[i] = (out[i - 1] * (n - 1) + dx[i]) / n
    return out


def supertrend(h, l, c, period, factor):
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2
    ub, lb = hl2 + factor * a, hl2 - factor * a
    N = len(c)
    fub, flb, trend = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    d = 1
    start = period
    if N <= start:
        return trend
    fub[start], flb[start] = ub[start], lb[start]
    trend[start] = flb[start]
    for i in range(start + 1, N):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d == 1 and c[i] < flb[i]:
            d = -1
        elif d == -1 and c[i] > fub[i]:
            d = 1
        trend[i] = flb[i] if d == 1 else fub[i]
    return trend


# ---------------------------------------------------------------- session
def session_mask(idx_utc, mode='cme'):
    if mode == '24x7':
        return np.ones(len(idx_utc), bool)
    ny = idx_utc.tz_convert('America/New_York')
    wd, hr = ny.dayofweek.values, ny.hour.values  # Mon=0 .. Sun=6
    m = np.zeros(len(ny), bool)
    m |= (wd == 6) & (hr >= 18)
    m |= (wd <= 3) & ((hr < 17) | (hr >= 18))
    m |= (wd == 4) & (hr < 17)
    return m


# ---------------------------------------------------------------- data prep (hp-independent parts cached)
class Prepared:
    def __init__(self, df1h, session='cme'):
        df = df1h.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        self.df = df
        self.idx = df.index
        self.o, self.h, self.l, self.c = (df[k].values.astype(float) for k in ('open', 'high', 'low', 'close'))
        self.sess = session_mask(df.index, session)
        # 4h candles aligned to UTC 00/04/08..., filtered to session by open time
        h4 = df.resample('4h', origin='epoch').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4 = h4[session_mask(h4.index, session)]
        self.h4 = h4
        # for 1h bar at t: anchor = 4h candles with start < floor4h(t)
        fl = df.index.floor('4h')
        self.a_pos = np.searchsorted(h4.index.values, fl.values, side='left') - 1  # last usable 4h row
        # session 1h candles, excluding current bar
        s_idx = np.where(self.sess)[0]
        self.s_idx = s_idx
        self.s_cnt_before = np.searchsorted(s_idx, np.arange(len(df)), side='left')  # session bars strictly before i
        sh, sl, sc = self.h[s_idx], self.l[s_idx], self.c[s_idx]
        self.s_atr = atr(sh, sl, sc, 14)
        self.sh, self.sl = sh, sl
        self._cache = {}

    def anchor_ind(self, hp):
        key = (hp['st_period'], hp['st_factor'], hp['ema4'])
        if key not in self._cache:
            H, L, C = self.h4.high.values, self.h4.low.values, self.h4.close.values
            self._cache[key] = dict(st=supertrend(H, L, C, hp['st_period'], hp['st_factor']),
                                    ema=ema(C, hp['ema4']) if hp['ema4'] > 0 else None)
        if 'atr' not in self._cache:
            H, L, C = self.h4.high.values, self.h4.low.values, self.h4.close.values
            self._cache['atr'] = atr(H, L, C, 14)
            self._cache['adx'] = adx(H, L, C, 14)
        return self._cache[key], self._cache['atr'], self._cache['adx']

    def donchian(self, n):
        k = ('don', n)
        if k not in self._cache:
            up = pd.Series(self.sh).rolling(n).max().values
            lo = pd.Series(self.sl).rolling(n).min().values
            # value usable at bar i = rolling over session bars [cnt-n, cnt)
            cnt = self.s_cnt_before
            ok = cnt >= n
            U = np.full(len(cnt), np.nan)
            Lo = np.full(len(cnt), np.nan)
            U[ok] = up[cnt[ok] - 1]
            Lo[ok] = lo[cnt[ok] - 1]
            self._cache[k] = (U, Lo)
        return self._cache[k]


# ---------------------------------------------------------------- backtest
def backtest(P: Prepared, hp=None, start=None, end=None, capital=10_000.0, fee=0.0002, slip=0.0,
             risk_pct=3.0, max_lev=4.0, leverage=5.0, funding_apr=0.0):
    hp = {**DEFAULT_HP, **(hp or {})}
    anc, A_atr, A_adx = P.anchor_ind(hp)
    st, em = anc['st'], anc['ema']
    U, Lo = P.donchian(int(hp['don_period']))
    o, h, l, c, idx = P.o, P.h, P.l, P.c, P.idx
    i0 = 0 if start is None else idx.searchsorted(_ts(start))
    i1 = len(idx) if end is None else idx.searchsorted(_ts(end))
    ap = P.a_pos

    bal = capital
    qty = 0.0  # signed
    entry = stop = tp = best = 0.0
    eq = np.empty(i1 - i0)
    trades = []
    ent_time = None
    fund_h = funding_apr / (365 * 24)

    def regime_at(i):
        a = ap[i]
        if a < 0 or np.isnan(st[a]) or st[a] <= 0:
            return 0
        d = 1 if c[i] > st[a] else -1
        if em is not None:
            e = em[a]
            if np.isnan(e):
                return 0
            if d == 1 and c[i] < e:
                return 0
            if d == -1 and c[i] > e:
                return 0
        return d

    def atr4(i):
        a = ap[i]
        v = A_atr[a] if a >= 0 else np.nan
        if not (v > 0):
            k = P.s_cnt_before[i] - 1
            v = P.s_atr[k] * 4 if k >= 0 else np.nan
        return v

    def close_pos(i, px, why):
        nonlocal bal, qty
        pnl = qty * (px - entry) - abs(qty) * px * fee
        bal += pnl
        trades.append(dict(entry_time=ent_time, exit_time=idx[i], side='long' if qty > 0 else 'short',
                           entry=entry, exit=px, qty=qty, pnl=pnl, why=why))
        qty = 0.0

    for j, i in enumerate(range(i0, i1)):
        # 1) intrabar exits for an open position (orders placed at previous close)
        if qty != 0:
            bal -= abs(qty) * c[i] * fund_h * (1 if qty > 0 else -1)  # longs pay funding when funding_apr>0
            if qty > 0:
                if l[i] <= stop:
                    close_pos(i, min(o[i], stop) * (1 - slip), 'stop')
            else:
                hit_s = h[i] >= stop
                hit_t = tp is not None and l[i] <= tp
                if hit_s:
                    close_pos(i, max(o[i], stop) * (1 + slip), 'stop')
                elif hit_t:
                    close_pos(i, min(o[i], tp), 'tp')
        # 2) at close: update_position
        if qty != 0:
            r = regime_at(i)
            if qty > 0:
                if hp['exit_on_flip'] == 1 and r != 1:
                    close_pos(i, c[i] * (1 - slip), 'flip')
                else:
                    best = max(best, c[i])
                    tr = best - atr4(i) * hp['trail_atr']
                    if stop < tr < c[i]:
                        stop = tr
            else:
                if hp['exit_on_flip'] == 1 and r != -1:
                    close_pos(i, c[i] * (1 + slip), 'flip')
                else:
                    best = min(best, c[i])
                    tr = best + atr4(i) * hp['s_trail']
                    if c[i] < tr < stop:
                        stop = tr
        # 3) entries
        elif P.sess[i] and not np.isnan(U[i]):
            r = regime_at(i)
            side = 0
            if r == 1 and not (hp['l_adx'] > 0 and not (A_adx[ap[i]] >= hp['l_adx'])) and c[i] > U[i]:
                side = 1
            elif r == -1 and not (hp['s_adx'] > 0 and not (A_adx[ap[i]] >= hp['s_adx'])) and c[i] < Lo[i]:
                side = -1
            if side != 0:
                a4 = atr4(i)
                if a4 > 0:
                    px = c[i] * (1 + slip * side)
                    stp = px - a4 * hp['atr_stop'] if side == 1 else px + a4 * hp['s_stop']
                    risk_size = (risk_pct / 100) * bal / abs(px - stp) * px
                    size = min(risk_size, bal * max_lev, bal * leverage * 0.95)
                    q = size / px * (1 - fee)
                    bal -= q * px * fee
                    qty = q * side
                    entry, stop, best, ent_time = px, stp, c[i], idx[i]
                    tp = (px - a4 * hp['s_tp']) if (side == -1 and hp['s_tp'] > 0) else None
        eq[j] = bal + (qty * (c[i] - entry) if qty != 0 else 0.0)
        if eq[j] <= 0:
            eq[j:] = 0
            break
    if qty != 0:
        close_pos(i1 - 1, c[i1 - 1], 'end')
    return pd.Series(eq, index=idx[i0:i1]), pd.DataFrame(trades)


# ---------------------------------------------------------------- metrics
def daily(eq):
    return eq.resample('1D').last().ffill()


def metrics(eq, trades, capital=10_000.0):
    d = daily(eq)
    r = d.pct_change().dropna()
    yrs = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    tot = d.iloc[-1] / capital - 1
    cagr = (d.iloc[-1] / capital) ** (1 / yrs) - 1 if d.iloc[-1] > 0 else -1
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
    dd = (d / d.cummax() - 1).min()
    n = len(trades)
    wr = (trades.pnl > 0).mean() if n else np.nan
    return dict(total=tot, cagr=cagr, sharpe=sharpe, maxdd=dd, trades=n, winrate=wr, years=yrs)


def buyhold(P, start=None, end=None):
    c = pd.Series(P.c, index=P.idx)
    if start is not None:
        c = c[c.index >= _ts(start)]
    if end is not None:
        c = c[c.index < _ts(end)]
    return c / c.iloc[0] * 10_000.0


def random_hp(rng):
    hp = {}
    for name, lo, hi, step, typ in HP_SPACE:
        n = int(round((hi - lo) / step))
        v = lo + step * rng.integers(0, n + 1)
        hp[name] = int(round(v)) if typ is int else float(v)
    return hp

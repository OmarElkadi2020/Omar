"""Stage 30: a model trained directly on Sharpe after costs, no label (PREREG_stage30_direct_profit_policy.md).
python -m tc.stage30 gradcheck|leak|tune|test   (run from research/crypto_trend_classifier)"""
import json
import sys
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from . import stage29 as s29
from .labels import oracle_labels
from .evaluate import supertrend

TRAIN = 'BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX DOT LTC BCH ATOM NEAR UNI FIL ETC XLM AAVE'.split()
EVAL = TRAIN[:10]
COINS = EVAL
COST = s29.COST
T0, T1 = pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2026-10-01', tz='UTC')
AGG = dict(open='first', high='max', low='min', close='last', volume='sum', quote_volume='sum', trades='sum',
           taker_buy_quote='sum')
GRID = list(itertools.product((8, 32), (1e-4, 1e-3, 1e-2), (0.001, 0.003)))
SEEDS = (0, 1, 2, 3, 4)
LR, PATIENCE, MAX_EP, VAL = 0.003, 30, 400, 0.2


# ------------------------------------------------------------------ model
def act(z, mode):
    return 1 / (1 + np.exp(-z)) if mode == 'lf' else np.tanh(z)


def dact(p, mode):
    return p * (1 - p) if mode == 'lf' else 1 - p * p


def pnl_and_grad(p, r, first, c):
    """p, r: concatenated coin sequences; first[t] = True at the first bar of each coin (no cost there).
    pnl_t = p_t r_t - c |p_t - p_{t-1}|.  Returns -Sharpe and d(-Sharpe)/dp."""
    dp = np.r_[0.0, np.diff(p)]
    dp[first] = 0.0
    s = np.sign(dp)
    pnl = p * r - c * np.abs(dp)
    N = len(pnl)
    mu, sd = pnl.mean(), pnl.std() + 1e-12
    g = (1 / sd - mu * (pnl - mu) / sd ** 3) / N           # dS/dpnl
    dS = g * (r - c * s)
    nxt = np.r_[s[1:], 0.0] * np.r_[(~first[1:]).astype(float), 0.0]
    dS += np.r_[g[1:], 0.0] * c * nxt
    return -mu / sd, -dS


class MLP:
    def __init__(self, nin, hidden, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(nin), (nin, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden) * 0.1
        self.b2 = 0.0

    def params(self):
        return [self.W1, self.b1, self.w2, np.array([self.b2])]

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return h, h @ self.w2 + self.b2

    def grads(self, X, h, dz, l2):
        gw2 = h.T @ dz + 2 * l2 * self.w2
        gb2 = dz.sum()
        dh = np.outer(dz, self.w2) * (1 - h * h)
        gW1 = X.T @ dh + 2 * l2 * self.W1
        gb1 = dh.sum(0)
        return [gW1, gb1, gw2, np.array([gb2])]

    def set(self, ps):
        self.W1, self.b1, self.w2, self.b2 = ps[0], ps[1], ps[2], float(ps[3][0])

    def copy(self):
        return [x.copy() for x in self.params()]


def loss_grad(m, X, r, first, mode, c, l2):
    h, z = m.forward(X)
    p = act(z, mode)
    L, dLdp = pnl_and_grad(p, r, first, c)
    dz = dLdp * dact(p, mode)
    return L + l2 * (np.sum(m.W1 ** 2) + np.sum(m.w2 ** 2)), m.grads(X, h, dz, l2)


def train_one(Xtr, rtr, ftr, Xva, rva, fva, mode, hidden, l2, c, seed):
    m = MLP(Xtr.shape[1], hidden, seed)
    ps = m.params()
    M = [np.zeros_like(x, float) for x in ps]
    V = [np.zeros_like(x, float) for x in ps]
    best, best_ps, wait = -np.inf, m.copy(), 0
    for ep in range(1, MAX_EP + 1):
        _, G = loss_grad(m, Xtr, rtr, ftr, mode, c, l2)
        ps = m.params()
        for i in range(len(ps)):
            M[i] = 0.9 * M[i] + 0.1 * G[i]
            V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
            ps[i] = ps[i] - LR * (M[i] / (1 - 0.9 ** ep)) / (np.sqrt(V[i] / (1 - 0.999 ** ep)) + 1e-8)
        m.set(ps)
        _, zv = m.forward(Xva)
        sv = -pnl_and_grad(act(zv, mode), rva, fva, COST)[0]   # validation at the real cost
        if sv > best + 1e-6:
            best, best_ps, wait = sv, m.copy(), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.set(best_ps)
    return m, ep, best


# ------------------------------------------------------------------ data
_D = None


def bars4h(d1):
    return d1.resample('4h', origin='epoch').agg(AGG).dropna(subset=['close'])


def flow_feats(df):
    F = {}
    for n in (6, 24, 72):
        F[f'tb_{n}'] = df.taker_buy_quote.rolling(n).sum() / df.quote_volume.rolling(n).sum().replace(0, np.nan) - 0.5
    lt = np.log(df.trades.replace(0, np.nan)).ffill()
    F['ntr_z'] = (lt - lt.rolling(180, min_periods=50).mean()) / lt.rolling(180, min_periods=50).std()
    return pd.DataFrame(F, index=df.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build(d1):
    df = bars4h(d1)
    X = s29.feats(df[['open', 'high', 'low', 'close', 'volume']]).join(flow_feats(df))
    return df, X


def data():
    global _D
    if _D is None:
        _D = {c: build(pd.read_parquet(f'data/binance_1h/{c}.parquet')) for c in TRAIN}
    return _D


def leak():
    bad = 0
    for c in ('BTC', 'SOL', 'AAVE'):
        d1 = pd.read_parquet(f'data/binance_1h/{c}.parquet')
        _, full = build(d1)
        for cut in pd.date_range('2021-03-01', '2026-06-01', periods=6, tz='UTC'):
            _, part = build(d1[d1.index < cut])
            part = part.iloc[:-1]            # the last 4h bar may be incomplete in the truncated 1h data
            ok = np.isclose(part.values, full.loc[part.index].values, rtol=1e-4, atol=1e-6, equal_nan=True)
            bad += int((~ok).sum())
    print('feature leak mismatches:', bad, '| features:', full.shape[1])
    return bad


def train_set(D, cut):
    """rows whose next return closes before the cut-off; standardisation from these rows only."""
    Xs, rs, fs, ts = [], [], [], []
    for c in TRAIN:
        df, X = D[c]
        nb = np.r_[df.index[1:], df.index[-1] + s29.STEP]          # open time of the next bar
        w = (nb + s29.STEP) <= cut                                  # next bar closed before the cut-off
        cl = df.close.values
        r = np.r_[cl[1:] / cl[:-1] - 1, 0.0]
        idx = np.flatnonzero(w)
        if len(idx) == 0:
            continue
        Xs.append(X.values[idx]); rs.append(r[idx]); ts.append(df.index[idx])
        f = np.zeros(len(idx), bool); f[0] = True; fs.append(f)
    X, r, f, t = np.vstack(Xs), np.concatenate(rs), np.concatenate(fs), np.concatenate(ts)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    return X, r, f, pd.DatetimeIndex(t), mu, sd


def norm(X, mu, sd):
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return np.clip(Z, -10, 10)


def fit(cut, mode, hidden, l2, c):
    D = data()
    X, r, f, t, mu, sd = train_set(D, cut)
    Z = norm(X, mu, sd)
    t0 = t.min()
    vcut = cut - (cut - t0) * VAL
    tr, va = np.asarray(t < vcut), np.asarray(t >= vcut)
    def firsts(mask):
        # first bar of each coin inside the subset
        out = np.zeros(mask.sum(), bool)
        sub = np.flatnonzero(mask)
        brk = np.r_[True, np.diff(sub) != 1] | f[sub]
        out[brk] = True
        return out
    models, info = [], []
    for seed in SEEDS:
        m, ep, best = train_one(Z[tr], r[tr], firsts(tr), Z[va], r[va], firsts(va), mode, hidden, l2, c, seed)
        models.append(m); info.append((ep, best))
    return models, mu, sd, info


def predict(models, mu, sd, X, mode):
    Z = norm(X, mu, sd)
    return np.mean([act(m.forward(Z)[1], mode) for m in models], axis=0)


# ------------------------------------------------------------------ commands
def gradcheck():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 7)); r = rng.normal(0, 0.02, 300); f = np.zeros(300, bool); f[[0, 150]] = True
    for mode in ('lf', 'ls'):
        m = MLP(7, 5, 1)
        L, G = loss_grad(m, X, r, f, mode, 0.003, 1e-3)
        ps = m.params(); err = 0
        for i in range(len(ps)):
            for j in [tuple(rng.integers(s) for s in ps[i].shape) for _ in range(5)]:
                q = [x.copy() for x in ps]; q[i][j] += 1e-6; m.set(q)
                L2 = loss_grad(m, X, r, f, mode, 0.003, 1e-3)[0]; m.set(ps)
                err = max(err, abs((L2 - L) / 1e-6 - G[i][j]) / (abs(G[i][j]) + 1e-6))
        print(mode, 'max relative gradient error', err)


def _dev_job(a):
    mode, hidden, l2, c, Y = a
    D = data()
    lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
    models, mu, sd, info = fit(lo, mode, hidden, l2, c)
    P = {k: predict(models, mu, sd, D[k][1].values, mode) for k in COINS}
    bar, daily = s29.portfolio(P, D, lo, hi, COST)
    to = np.mean([np.abs(np.diff(P[k][s29.window(D[k][0].index, lo, hi)])).sum() for k in COINS]) / 1.0
    return dict(mode=mode, hidden=hidden, l2=l2, c_train=c, fold=Y, sharpe=s29.sharpe_d(daily),
                cagr=s29.money(bar, daily)['cagr'], turnover_y=float(to), epochs=[e for e, _ in info])


def tune():
    jobs = [(mode, h, l2, c, Y) for mode in ('lf', 'ls') for (h, l2, c) in GRID for Y in s29.FOLDS]
    D = data()
    with Pool(4) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_dev_job, jobs)):
            rows.append(row)
            print(i + 1, '/', len(jobs), row['mode'], row['hidden'], row['l2'], row['c_train'], row['fold'],
                  round(row['sharpe'], 3), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('results_stage30_dev.csv', index=False)
    G = R.groupby(['mode', 'hidden', 'l2', 'c_train']).sharpe.mean().reset_index()
    fz = {}
    for mode in ('lf', 'ls'):
        g = G[G['mode'] == mode].sort_values('sharpe', ascending=False).iloc[0]
        fz[mode] = dict(hidden=int(g.hidden), l2=float(g.l2), c_train=float(g.c_train), dev_sharpe=float(g.sharpe))
    st29 = tune29()
    for Y in s29.FOLDS:
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        P = {c: (supertrend(D[c][0], *s29.ST) == 1).astype(int) for c in EVAL}
        st29[f'ST_dev_sharpe_{Y}'] = s29.sharpe_d(s29.portfolio(P, D, lo, hi, COST)[1])
    json.dump(dict(choice=fz, st29=st29, seeds=list(SEEDS), lr=LR, patience=PATIENCE, max_epochs=MAX_EP, val=VAL),
              open('prereg/FROZEN_stage30.json', 'w'), indent=1)
    print(json.dumps(fz, indent=1))


def tune29():
    """stage-29 procedure (dev 2022-23, portfolio Sharpe, long/flat) for EQUAL and WEIGHTED-24h on this data."""
    D = data()
    fz = {}
    for kind in ('EQUAL', 'WEIGHTED-24h'):
        rows = []
        for Y in s29.FOLDS:
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
            m = s29.fit(D, lo, kind, TRAIN)
            Pu = {c: m.predict(D[c][1].values.astype(np.float32)) for c in EVAL}
            for span, th in s29.CONV:
                P = {c: s29.state(Pu[c], span, th) for c in EVAL}
                rows.append(dict(span=span, th=th, fold=Y, sharpe=s29.sharpe_d(s29.portfolio(P, D, lo, hi, COST)[1])))
        g = pd.DataFrame(rows).groupby(['span', 'th']).sharpe.mean().sort_values(ascending=False)
        fz[kind] = dict(span=int(g.index[0][0]), th=float(g.index[0][1]), dev_sharpe=float(g.iloc[0]))
        print(kind, fz[kind], flush=True)
    return fz


def _test_job(a):
    mode, Y, ch = a
    D = data()
    cut = pd.Timestamp(f'{Y}-01-01', tz='UTC')
    models, mu, sd, info = fit(cut, mode, ch['hidden'], ch['l2'], ch['c_train'])
    return mode, Y, {k: predict(models, mu, sd, D[k][1].values, mode) for k in COINS}, info


def test():
    D = data()
    fz = json.load(open('prereg/FROZEN_stage30.json'))['choice']
    f29 = json.load(open('prereg/FROZEN_stage30.json'))['st29']
    with Pool(4) as pool:
        out = pool.map(_test_job, [(mode, Y, fz[mode]) for mode in ('lf', 'ls') for Y in s29.TEST_YEARS])
    POL = {}
    for mode in ('lf', 'ls'):
        P = {c: np.full(len(D[c][0]), np.nan) for c in COINS}
        for m_, Y, pr, info in out:
            if m_ != mode:
                continue
            for c in COINS:
                idx = D[c][0].index
                w = (idx >= pd.Timestamp(f'{Y}-01-01', tz='UTC')) & (idx < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))
                P[c][w] = pr[c][w]
            print(mode, Y, 'epochs', [e for e, _ in info], flush=True)
        POL[mode] = P
    # stage-29 classifiers with frozen settings
    CL = {}
    for kind in ('EQUAL', 'WEIGHTED-24h'):
        Pu = {c: np.full(len(D[c][0]), np.nan) for c in COINS}
        for Y in s29.TEST_YEARS:
            cut, nxt = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
            m = s29.fit(D, cut, kind, TRAIN)
            for c in COINS:
                idx = D[c][0].index
                w = (idx >= (cut if Y > s29.TEST_YEARS[0] else idx[0])) & (idx < nxt)
                Pu[c][w] = m.predict(D[c][1].values[w].astype(np.float32))
        CL[kind] = {c: s29.state(Pu[c], f29[kind]['span'], f29[kind]['th']) for c in COINS}
    ST = {c: (supertrend(D[c][0], *s29.ST) == 1).astype(int) for c in COINS}
    LAB = {c: (oracle_labels(D[c][0].close.values, 1.0) == 1).astype(int) for c in COINS}

    def strategies(mode):
        pos = lambda S: {c: s29.positions(S[c], mode) for c in COINS}
        disc = {c: ((POL[mode][c] >= 0.5).astype(float) if mode == 'lf' else np.sign(POL[mode][c])) for c in COINS}
        S = {'POLICY': {c: np.nan_to_num(POL[mode][c]) for c in COINS}, 'POLICY discretised': disc,
             'EQUAL (st.29)': pos(CL['EQUAL']), 'WEIGHTED-24h (st.29)': pos(CL['WEIGHTED-24h']),
             'SuperTrend(48,5)': pos(ST), 'B1 label (hindsight)': pos(LAB)}
        if mode == 'lf':
            S['buy & hold'] = {c: np.ones(len(D[c][0])) for c in COINS}
        return S

    rows, coin_rows, year_rows, daily = [], [], [], {}
    for mode in ('lf', 'ls'):
        S = strategies(mode)
        for cost in (COST, 2 * COST):
            for name, P in S.items():
                bar, d = s29.portfolio(P, D, T0, T1, cost)
                daily[(name, mode, cost)] = d
                VL, TO, Tall = [], [], []
                for c in COINS:
                    df = D[c][0]
                    w = s29.window(df.index, T0, T1)
                    cl = df.close.values[w]
                    v = s29.vs_label(P[c][w], s29.positions(LAB[c], mode)[w], cl, cost)
                    VL.append(v)
                    TO.append(np.abs(np.diff(P[c][w])).sum() / (w.sum() / s29.BPY))
                    pd_ = P[c][w] if name != 'POLICY' else S['POLICY discretised'][c][w]
                    Tall.append(s29.trades(pd_, cl, cost))
                    if cost == COST:
                        cb, cd = s29.portfolio({c: P[c]}, {c: D[c]}, T0, T1, cost)
                        coin_rows.append(dict(strategy=name, mode=mode, coin=c, **s29.money(cb, cd), **v))
                yrs = len(bar) / s29.BPY
                rows.append(dict(strategy=name, mode=mode, cost=cost, **s29.money(bar, d),
                                 turnover_y=float(np.mean(TO)), **pd.DataFrame(VL).mean().to_dict(),
                                 **s29.trade_stats(np.concatenate(Tall), yrs * len(COINS)),
                                 mean_pos=float(np.mean([np.mean(P[c][s29.window(D[c][0].index, T0, T1)]) for c in COINS]))))
                if cost == COST:
                    for Y in s29.TEST_YEARS:
                        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), min(pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'), s29.T1)
                        b2, d2 = s29.portfolio(P, D, lo, hi, cost)
                        year_rows.append(dict(strategy=name, mode=mode, year=Y, **s29.money(b2, d2)))
    R = pd.DataFrame(rows)
    R.to_csv('results_stage30_test.csv', index=False)
    C = pd.DataFrame(coin_rows); C.to_csv('results_stage30_per_coin.csv', index=False)
    pd.DataFrame(year_rows).to_csv('results_stage30_years.csv', index=False)

    st = {}
    for mode in ('lf', 'ls'):
        for b in ('SuperTrend(48,5)', 'EQUAL (st.29)', 'WEIGHTED-24h (st.29)', 'buy & hold'):
            if b == 'buy & hold' and mode == 'ls':
                continue
            for cost in (COST, 2 * COST):
                x, y = daily[('POLICY', mode, cost)], daily[(b, mode, cost)]
                j = x.index.intersection(y.index)
                d0, p, lo5, hi95 = s29.stationary_bootstrap_diff(x[j].values, y[j].values)
                st[f'{mode} cost {cost}: POLICY - {b}'] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95])
        bh = daily[('buy & hold', 'lf', COST)]
        for name in ('POLICY', 'SuperTrend(48,5)', 'EQUAL (st.29)'):
            x = daily[(name, mode, COST)]
            j = x.index.intersection(bh.index)
            al, t, beta = s29.nw_alpha(x[j].values, bh[j].values)
            st[f'{mode}: alpha vs B&H: {name}'] = dict(alpha_ann=al, t=t, beta=beta)
        cw = C[C['mode'] == mode].pivot(index='coin', columns='strategy', values='sharpe')
        n = int((cw['POLICY'] > cw['SuperTrend(48,5)']).sum())
        k = st[f'{mode} cost {COST}: POLICY - SuperTrend(48,5)']
        st[f'PRIMARY {mode}'] = dict(sharpe_diff=k['sharpe_diff'], p=k['p_one_sided'], coins_better=n,
                                     met=bool(k['sharpe_diff'] > 0 and k['p_one_sided'] < 0.025 and n >= 8))
    st['PRIMARY met (either mode)'] = st['PRIMARY lf']['met'] or st['PRIMARY ls']['met']
    json.dump(st, open('results_stage30_stats.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(R[R.cost == COST][['strategy', 'mode', 'cagr', 'sharpe', 'maxdd', 'turnover_y', 'mean_pos', 'trades_y', 'win',
                             'capture', 'missed', 'wrong', 'extra_cost', 'acc_money']].round(3))
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    {'gradcheck': gradcheck, 'leak': leak, 'tune': tune, 'test': test}[sys.argv[1]]()

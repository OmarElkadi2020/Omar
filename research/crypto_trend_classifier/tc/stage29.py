"""Stage 29: profit-weighted training vs equal weights, judged in money (PREREG_stage29_profit_weighted_money_eval.md).
python -m tc.stage29 leak|tune|test   (run from research/crypto_trend_classifier)"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import core_features, resample, join_by_close, HTF_KEEP
from .labels import oracle_labels
from .s12lib import conv_h, ewm_score
from .evaluate import supertrend

COINS = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL']
STEP = pd.Timedelta('4h')
BPY = 2190
EMB = 24
FOLDS = (2022, 2023)
TEST_YEARS = (2024, 2025, 2026)
T0, T1 = pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2026-09-24', tz='UTC')
COST = 0.001
ST = (48, 5.0)
P25 = dict(leaves=11, lr=0.010972939616962498, trees=150, min_leaf=525, ff=0.4311261058631236,
           bf=0.9399786467375328, l2=75.232994934394)
CONV = [(span, th) for span in (1, 3, 6, 12, 24) for th in (0.0, 0.1, 0.2, 0.3, 0.4)]
KINDS = ('EQUAL', 'WEIGHTED', 'WEIGHTED-24h')


# ------------------------------------------------------------------ data and features
def raw():
    s = open('trend_compass.html').read()
    a = s.find('application/json">') + 18
    D = json.loads(s[a:s.find('</script>', a)])
    out = {}
    for c in COINS:
        d = D[f'{c}USDT|4h']
        idx = pd.to_datetime(d['t'], unit='s', utc=True)
        out[c] = pd.DataFrame({k: np.asarray(d[k[0]], float) for k in ('open', 'high', 'low', 'close')}, index=idx)
    return out


def feats(df):
    X = core_features(df)
    for m in (4, 24):
        hs = STEP * m
        hf = core_features(resample(df, hs))[HTF_KEEP]
        X = X.join(join_by_close(df.index, STEP, hf, hs, f'h{m}_'))
    return X


def load():
    R = raw()
    return {c: (R[c], feats(R[c])) for c in COINS}


def leak():
    R = raw()
    bad = 0
    for c in COINS:
        full = feats(R[c])
        for cut in pd.date_range('2021-03-01', '2026-06-01', periods=6, tz='UTC'):
            part = feats(R[c][R[c].index < cut])
            a, b = part.values, full.loc[part.index].values
            ok = np.isclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True)
            bad += int((~ok).sum())
    print('feature leak mismatches:', bad)
    return bad


# ------------------------------------------------------------------ training
def train_rows(df, X, cut):
    m = df.index < cut
    c = df.close.values[m]
    lab, fin = oracle_labels(c, 1.0, return_final=True)
    keep = max(fin - EMB, 0)
    assert keep + 6 <= len(c) and df.index[m][keep - 1] < cut   # weights and rows stay before the cut-off
    lr = np.log(c)
    w1 = np.abs(lr[1:keep + 1] - lr[:keep])
    w24 = np.abs(lr[6:keep + 6] - lr[:keep])
    return X.values[m][:keep].astype(np.float32), (lab[:keep] == 1).astype(int), w1, w24


def fit(D, cut, kind):
    Xs, ys, ws = [], [], []
    for c in COINS:
        df, X = D[c]
        x, y, w1, w24 = train_rows(df, X, cut)
        Xs.append(x); ys.append(y); ws.append(w1 if kind == 'WEIGHTED' else w24)
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    w = None if kind == 'EQUAL' else w / w.mean()
    p = dict(objective='binary', learning_rate=P25['lr'], num_leaves=P25['leaves'], min_data_in_leaf=P25['min_leaf'],
             feature_fraction=P25['ff'], bagging_fraction=P25['bf'], bagging_freq=1, lambda_l2=P25['l2'],
             verbose=-1, num_threads=4, seed=0, deterministic=True)
    cols = list(D[COINS[0]][1].columns)
    return lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=cols), num_boost_round=P25['trees'])


def state(p_up, span, th):
    s = ewm_score(2 * np.asarray(p_up) - 1, span)
    st = conv_h(s, th, -th)
    return (pd.Series(st).replace(0, np.nan).ffill().fillna(1).values > 0).astype(int)


# ------------------------------------------------------------------ money
def coin_pnl(pos, c, cost):
    """pos[t] held over (t, t+1]; returns simple and log P&L per bar (last bar 0) and cost per bar."""
    r = np.r_[c[1:] / c[:-1] - 1, 0.0]
    dp = np.abs(np.diff(pos, prepend=0.0))
    cst = cost * dp
    simple = pos * r - cst
    return simple, cst


def window(idx, lo, hi):
    return (idx >= lo) & (idx < hi)


def portfolio(P, D, lo, hi, cost):
    """P: coin -> position array over full index. Equal weight over coins alive in the window. Daily simple returns."""
    cols = {}
    for c in P:
        df = D[c][0]
        w = window(df.index, lo, hi)
        if not w.any():
            continue
        pos = np.asarray(P[c], float)[w]
        cl = df.close.values[w]
        s, _ = coin_pnl(pos, cl, cost)
        cols[c] = pd.Series(s, df.index[w])
    B = pd.DataFrame(cols)
    bar = B.mean(axis=1, skipna=True)
    return bar, (1 + bar).resample('1D').prod() - 1


def sharpe_d(d):
    d = np.asarray(d)
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0


def money(bar, daily):
    eq = (1 + bar).cumprod()
    yrs = len(bar) / BPY
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return dict(cagr=float(cagr), vol=float(daily.std() * np.sqrt(365)), sharpe=sharpe_d(daily), maxdd=float(dd),
                calmar=float(cagr / abs(dd)) if dd < 0 else np.nan, total=float(eq.iloc[-1] - 1))


def trades(pos, c, cost):
    pos = np.asarray(pos, float)
    lr = np.r_[np.diff(np.log(c)), 0.0]
    ch = np.flatnonzero(np.diff(pos) != 0) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(pos)]
    out = []
    for s, e in zip(st, en):
        if pos[s] == 0:
            continue
        out.append(float(np.sum(pos[s:e] * lr[s:e]) + 2 * np.log(1 - cost * abs(pos[s]))))
    return np.array(out)


def trade_stats(T, years):
    if len(T) == 0:
        return dict(trades_y=0, win=np.nan, avg_win=np.nan, avg_loss=np.nan, pf=np.nan, expect=np.nan, top10=np.nan)
    w, l = T[T > 0], T[T <= 0]
    srt = np.sort(T)[::-1]
    return dict(trades_y=len(T) / years, win=float((T > 0).mean()),
                avg_win=float(np.expm1(w.mean())) if len(w) else np.nan,
                avg_loss=float(np.expm1(l.mean())) if len(l) else np.nan,
                pf=float(w.sum() / -l.sum()) if l.sum() < 0 else np.inf, expect=float(np.expm1(T.mean())),
                top10=float(srt[:max(1, len(T) // 10)].sum() / T.sum()) if T.sum() > 0 else np.nan)


def vs_label(pos, lab_pos, c, cost):
    """log-return per year: label P&L - strategy P&L = missed + wrong + extra cost (exact)."""
    lr = np.r_[np.diff(np.log(c)), 0.0]
    yrs = len(c) / BPY
    lc = lambda p: np.sum(-np.log(1 - cost * np.abs(np.diff(p, prepend=0.0))))
    S, L = np.asarray(pos, float), np.asarray(lab_pos, float)
    gap = (L - S) * lr
    missed = gap[(L > S)].sum()          # label more long than strategy
    wrong = gap[(L < S)].sum()           # strategy more long than label
    extra = lc(S) - lc(L)
    pl_l = np.sum(L * lr) - lc(L)
    pl_s = np.sum(S * lr) - lc(S)
    a = np.abs(lr)
    return dict(pl_label=pl_l / yrs, pl_strat=pl_s / yrs, capture=pl_s / pl_l if pl_l else np.nan,
                missed=missed / yrs, wrong=wrong / yrs, extra_cost=extra / yrs,
                acc=float((S == L).mean()), acc_money=float((a * (S == L)).sum() / a.sum()))


def stationary_bootstrap_diff(a, b, n=10000, block=20, seed=0):
    """paired stationary bootstrap of Sharpe(a) - Sharpe(b); returns diff, one-sided p (diff <= 0), 90% CI."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    T = len(a)
    d0 = sharpe_d(a) - sharpe_d(b)
    t = np.arange(T)
    new = rng.random((n, T)) < 1.0 / block
    new[:, 0] = True
    bstart = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    start = rng.integers(T, size=(n, T))[np.arange(n)[:, None], bstart]
    idx = (start + t - bstart) % T
    sh = lambda x: x.mean(1) / x.std(1) * np.sqrt(365)
    out = sh(a[idx]) - sh(b[idx])
    return d0, float((out <= 0).mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def nw_alpha(y, x, lags=10):
    y, x = np.asarray(y), np.asarray(x)
    X = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        g = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += (1 - L / (lags + 1)) * (g + g.T)
    V = XtXi @ S @ XtXi
    return float(beta[0] * 365), float(beta[0] / np.sqrt(V[0, 0])), float(beta[1])


def positions(state_up, mode):
    s = np.asarray(state_up)
    return s.astype(float) if mode == 'lf' else np.where(s == 1, 1.0, -1.0)


# ------------------------------------------------------------------ dev tuning
def tune():
    D = load()
    rows = []
    for kind in KINDS:
        for Y in FOLDS:
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
            m = fit(D, lo, kind)
            Pu = {c: m.predict(D[c][1].values.astype(np.float32)) for c in COINS}
            for span, th in CONV:
                P = {c: state(Pu[c], span, th) for c in COINS}
                bar, daily = portfolio(P, D, lo, hi, COST)
                rows.append(dict(kind=kind, fold=Y, span=span, th=th, sharpe=sharpe_d(daily),
                                 cagr=money(bar, daily)['cagr']))
            print(kind, Y, 'done', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('results_stage29_dev.csv', index=False)
    G = R.groupby(['kind', 'span', 'th']).sharpe.mean().reset_index()
    fz = {}
    for kind in KINDS:
        g = G[G.kind == kind].sort_values('sharpe', ascending=False).iloc[0]
        fz[kind] = dict(span=int(g.span), th=float(g.th), dev_sharpe=float(g.sharpe))
    # SuperTrend on the same dev folds, for reference only
    for Y in FOLDS:
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        P = {c: (supertrend(D[c][0], *ST) == 1).astype(int) for c in COINS}
        fz[f'ST_dev_sharpe_{Y}'] = sharpe_d(portfolio(P, D, lo, hi, COST)[1])
    json.dump(dict(conv=fz, lgb=P25, cost=COST), open('prereg/FROZEN_stage29.json', 'w'), indent=1)
    print(json.dumps(fz, indent=1))


# ------------------------------------------------------------------ test (run once)
def test():
    D = load()
    fz = json.load(open('prereg/FROZEN_stage29.json'))['conv']
    # yearly refits; each year's bars predicted by the model fitted on 1 January of that year
    S = {}
    for kind in KINDS:
        Pu = {c: np.full(len(D[c][0]), np.nan) for c in COINS}
        for Y in TEST_YEARS:
            cut, nxt = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
            m = fit(D, cut, kind)
            for c in COINS:
                idx = D[c][0].index
                w = (idx >= (cut if Y > TEST_YEARS[0] else idx[0])) & (idx < nxt)   # first model also warms up the EWM
                Pu[c][w] = m.predict(D[c][1].values[w].astype(np.float32))
            print(kind, Y, 'fitted', flush=True)
        S[kind] = {c: state(Pu[c], fz[kind]['span'], fz[kind]['th']) for c in COINS}
    S['SuperTrend(48,5)'] = {c: (supertrend(D[c][0], *ST) == 1).astype(int) for c in COINS}
    S['buy & hold'] = {c: np.ones(len(D[c][0]), int) for c in COINS}
    S['B1 label (hindsight)'] = {c: (oracle_labels(D[c][0].close.values, 1.0) == 1).astype(int) for c in COINS}

    rows, coin_rows, year_rows, daily = [], [], [], {}
    for mode in ('lf', 'ls'):
        for cost in (COST, 2 * COST):
            for name, St in S.items():
                if name == 'buy & hold' and mode == 'ls':
                    continue
                P = {c: positions(St[c], mode) for c in COINS}
                bar, d = portfolio(P, D, T0, T1, cost)
                daily[(name, mode, cost)] = d
                Tall, VL = [], []
                for c in COINS:
                    df = D[c][0]
                    w = window(df.index, T0, T1)
                    cl = df.close.values[w]
                    Tall.append(trades(P[c][w], cl, cost))
                    lab = positions(S['B1 label (hindsight)'][c], mode)[w]
                    v = vs_label(P[c][w], lab, cl, cost)
                    VL.append(v)
                    if cost == COST:
                        cb, cd = portfolio({c: P[c]}, {c: D[c]}, T0, T1, cost)
                        coin_rows.append(dict(strategy=name, mode=mode, coin=c, **money(cb, cd), **v,
                                              **trade_stats(Tall[-1], w.sum() / BPY)))
                yrs = len(bar) / BPY
                V = pd.DataFrame(VL).mean().to_dict()
                rows.append(dict(strategy=name, mode=mode, cost=cost, **money(bar, d),
                                 **trade_stats(np.concatenate(Tall), yrs * len(COINS)), **V,
                                 time_in_mkt=float(np.mean([np.mean(P[c][window(D[c][0].index, T0, T1)] != 0) for c in COINS]))))
                if cost == COST:
                    for Y in TEST_YEARS:
                        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), min(pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'), T1)
                        b2, d2 = portfolio(P, D, lo, hi, cost)
                        year_rows.append(dict(strategy=name, mode=mode, year=Y, **money(b2, d2)))
    R = pd.DataFrame(rows)
    R.to_csv('results_stage29_test.csv', index=False)
    pd.DataFrame(coin_rows).to_csv('results_stage29_per_coin.csv', index=False)
    pd.DataFrame(year_rows).to_csv('results_stage29_years.csv', index=False)

    # statistics
    st = {}
    pairs = [('WEIGHTED', 'EQUAL'), ('WEIGHTED-24h', 'EQUAL'), ('WEIGHTED', 'SuperTrend(48,5)'),
             ('EQUAL', 'SuperTrend(48,5)'), ('SuperTrend(48,5)', 'buy & hold'), ('WEIGHTED', 'buy & hold')]
    for mode in ('lf', 'ls'):
        for a, b in pairs:
            if b == 'buy & hold' and mode == 'ls':
                continue
            x, y = daily[(a, mode, COST)], daily[(b, mode, COST)]
            j = x.index.intersection(y.index)
            d0, p, lo5, hi95 = stationary_bootstrap_diff(x[j].values, y[j].values)
            st[f'{mode}: {a} - {b}'] = dict(sharpe_diff=d0, p_one_sided=p, ci90=[lo5, hi95])
        bh = daily[('buy & hold', 'lf', COST)]
        for name in S:
            if name in ('buy & hold',):
                continue
            x = daily[(name, mode, COST)]
            j = x.index.intersection(bh.index)
            al, t, beta = nw_alpha(x[j].values, bh[j].values)
            st[f'{mode}: alpha vs B&H: {name}'] = dict(alpha_ann=al, t=t, beta=beta)
    C = pd.DataFrame(coin_rows)
    cw = C[(C['mode'] == 'lf')].pivot(index='coin', columns='strategy', values='sharpe')
    st['lf: coins where WEIGHTED Sharpe > EQUAL'] = int((cw['WEIGHTED'] > cw['EQUAL']).sum())
    lf = R[(R['mode'] == 'lf') & (R.cost == COST)].set_index('strategy')
    k = st['lf: WEIGHTED - EQUAL']
    st['PRIMARY'] = dict(c1_sharpe_diff_p_lt_005=bool(k['sharpe_diff'] > 0 and k['p_one_sided'] < 0.05),
                         c2_capture_higher=bool(lf.loc['WEIGHTED', 'capture'] > lf.loc['EQUAL', 'capture']),
                         c3_coins_ge_4=bool(st['lf: coins where WEIGHTED Sharpe > EQUAL'] >= 4))
    st['PRIMARY']['met'] = all(st['PRIMARY'].values())
    json.dump(st, open('results_stage29_stats.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(R[R.cost == COST][['strategy', 'mode', 'cagr', 'sharpe', 'maxdd', 'trades_y', 'win', 'pf', 'top10',
                             'capture', 'missed', 'wrong', 'extra_cost', 'acc', 'acc_money', 'time_in_mkt']].round(3))
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    {'leak': leak, 'tune': tune, 'test': test}[sys.argv[1]]()

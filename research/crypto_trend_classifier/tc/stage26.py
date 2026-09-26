"""Stage 26: meta-labeling SuperTrend as a 1h long trend filter (PREREG_stage26_metalabel_supertrend_1h.md).
python -m tc.stage26 build|leak|events|tune|test"""
import json
import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.metrics import matthews_corrcoef
from .stage14 import frame as frame14, market as market14
from .labels import oracle_labels
from .evaluate import supertrend
from .features import _atr
from .s12lib import lag_metrics
from .stage16 import CD, TZ
from . import stage25 as s25

EVAL, EXTRA, COINS = s25.EVAL, s25.EXTRA, s25.COINS
S26 = 'bnc/s26'
OUT = f'{CD}/s26'
FOLDS = (2022, 2023)
T0, T1 = pd.Timestamp('2024-01-01', tz=TZ), pd.Timestamp('2026-09-01', tz=TZ)
EMB, CAP, TRIM = 24, 720, 240
RT_COST, COST = 0.002, 0.001
PRIMS = [(p, m) for p in (10, 14, 24, 48) for m in (2, 3, 4)]
ST_GRID = [(p, m) for p in (7, 10, 14, 24, 48) for m in (1.5, 2, 3, 4, 5)]
TAUS = [round(x, 2) for x in np.arange(0.20, 0.701, 0.05)]
TRUNC = None                       # leak test: cut all raw data at this time
H4 = pd.Timedelta('4h')
H1 = pd.Timedelta('1h')


# ------------------------------------------------------------------ raw data
def spot1h(c):
    d = pd.read_pickle(f'{S26}/{c}_spot1h.pkl')
    d.index.name = 'timestamp'
    if TRUNC is not None:
        d = d[d.index + H1 <= TRUNC]
    return d


def loader1h(c):
    return spot1h(c)[['open', 'high', 'low', 'close', 'volume']].astype(float)


def h4frame(c):
    g = s25.load(c)
    if TRUNC is not None:
        g = g[g.index + H4 <= TRUNC]           # stage-25 4h rows are causal (stage-25 leak test)
    return g


def flow1h(c, idx):
    sp = spot1h(c).reindex(idx)
    F = pd.DataFrame(index=idx)
    sh = sp.tb_base / sp.volume.replace(0, np.nan)
    for n in (6, 24, 72):
        F[f'tb_share_{n}'] = sh.rolling(n, min_periods=n // 2).mean()
    lt = np.log1p(sp.trades)
    F['trades_z'] = (lt - lt.rolling(720, min_periods=240).mean()) / lt.rolling(720, min_periods=240).std()
    fn = f'{S26}/{c}_perp1h.pkl'
    if os.path.exists(fn):
        pp = pd.read_pickle(fn)
        if TRUNC is not None:
            pp = pp[pp.index + H1 <= TRUNC]
        pp = pp.reindex(idx)
        F['perp_tb_share_24'] = (pp.tb_base / pp.volume.replace(0, np.nan)).rolling(24, min_periods=12).mean()
        F['perp_spot_vol'] = np.log(pp.qvol.rolling(24, min_periods=12).mean() / sp.qvol.rolling(24, min_periods=12).mean())
    return F


def align4h(g, idx1):
    """row j of the 4h frame for every 1h bar: the last 4h bar whose close <= the 1h bar close."""
    j = np.searchsorted((g.index + H4).values, (idx1 + H1).values, side='right') - 1
    return j


def st_state(f, p, m):
    try:
        return (np.asarray(supertrend(f, p, float(m))) == 1).astype(np.int8)
    except ZeroDivisionError:
        a = _atr(f, p).values
        ok = np.isfinite(a) & (a > 0)
        g = f.copy()
        g.loc[~ok, ['high', 'low']] = np.c_[g.close[~ok] * 1.0001, g.close[~ok] * 0.9999]
        return (np.asarray(supertrend(g, p, float(m))) == 1).astype(np.int8)


def build_one(args):
    c, mk = args
    f = frame14(loader1h, c, '1h', mk)
    idx = f.index
    base = f[['open', 'high', 'low', 'close']]
    X1 = f.drop(columns=['open', 'high', 'low', 'close']).add_prefix('h1_')
    fl = flow1h(c, idx).add_prefix('h1flow_')
    g = h4frame(c)
    j = align4h(g, idx)
    cols4 = s25.feat_cols(g)
    v = g[cols4].values.astype(np.float32)
    A = np.where(j[:, None] >= 0, v[np.maximum(j, 0)], np.nan)
    X4 = pd.DataFrame(A, index=idx, columns=['h4_' + x for x in cols4])
    st4 = s25.indicator_state(g, 'SuperTrend', dict(p=48, m=4)).astype(float)
    X4['h4_st48_4'] = np.where(j >= 0, st4[np.maximum(j, 0)], np.nan)
    out = pd.concat([base, X1, fl, X4], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return c, out.astype({k: np.float32 for k in out.columns if k not in ('open', 'high', 'low', 'close')})


def market1h():
    return market14({c: loader1h(c).close for c in COINS})


def build():
    os.makedirs(OUT, exist_ok=True)
    mk = market1h()
    with Pool(4) as pool:
        for c, out in pool.imap_unordered(build_one, [(c, mk) for c in COINS]):
            out.to_pickle(f'{OUT}/{c}.pkl')
            print('built', c, out.shape, flush=True)


def load(c):
    return pd.read_pickle(f'{OUT}/{c}.pkl')


# ------------------------------------------------------------------ events
def event_features(f, st, p):
    """context of each up-flip of the primary; uses bars <= i only."""
    c = f.close.values
    n = len(c)
    atr = _atr(f, p).values
    flip = np.r_[0, np.diff(st.astype(int))]
    ups = np.flatnonzero(flip == 1)
    dns = np.flatnonzero(flip == -1)
    lc = np.log(c)
    nfl = pd.Series(np.abs(flip)).rolling(168, min_periods=1).sum().values
    upsh = pd.Series(st.astype(float)).rolling(168, min_periods=24).mean().values
    rows = []
    for k, i in enumerate(ups):
        pu = ups[k - 1] if k > 0 else -1
        pd_ = dns[np.searchsorted(dns, i) - 1] if np.searchsorted(dns, i) > 0 else -1     # last down-flip before i
        prev_ret = lc[pd_] - lc[pu] if (pu >= 0 and pd_ > pu) else np.nan
        rows.append(dict(ev_atr_pct=atr[i] / c[i], ev_ret_p=(lc[i] - lc[max(i - p, 0)]) / max(atr[i] / c[i], 1e-9),
                         ev_since_prev=np.log1p(i - pu) if pu >= 0 else np.nan,
                         ev_prev_ret=prev_ret, ev_prev_ret_atr=prev_ret / max(atr[i] / c[i], 1e-9),
                         ev_prev_len=np.log1p(pd_ - pu) if (pu >= 0 and pd_ > pu) else np.nan,
                         ev_down_len=np.log1p(i - pd_) if pd_ >= 0 else np.nan,
                         ev_flips_168=nfl[i], ev_up_168=upsh[i]))
    return ups, pd.DataFrame(rows, index=ups)


def segment_ends(st, ups):
    """bar index of the down-flip ending each up-segment (len(st) if still open)."""
    dn = np.flatnonzero(np.r_[0, np.diff(st.astype(int))] == -1)
    k = np.searchsorted(dn, ups, side='right')
    return np.where(k < len(dn), dn[np.minimum(k, len(dn) - 1)], len(st))


def events():
    """per primary: event table (features + bookkeeping) for all coins."""
    os.makedirs(f'{OUT}/ev', exist_ok=True)
    for c in COINS:
        f = load(c)
        feat = [x for x in f.columns if x not in ('open', 'high', 'low', 'close')]
        c_ = f.close.values
        for p, m in PRIMS:
            st = st_state(f, p, m)
            ups, E = event_features(f, st, p)
            end = segment_ends(st, ups)
            lab_end = np.minimum(end, ups + CAP)                       # label horizon
            finished = lab_end < len(st)
            le = np.minimum(lab_end, len(st) - 1)
            E['coin'] = c
            E['i'] = ups
            E['t'] = f.index[ups]
            E['end'] = end
            E['lab_end'] = lab_end
            E['lab_end_t'] = np.where(finished, f.index[le].asi8, np.iinfo(np.int64).max)
            E['ret'] = np.log(c_[le] / c_[ups])
            X = f[feat].iloc[ups].reset_index(drop=True)
            X.index = E.index
            E = pd.concat([E, X], axis=1)
            fn = f'{OUT}/ev/{p}_{m}_{c}.pkl'
            E.to_pickle(fn)
        print('events', c, flush=True)


def load_events(p, m, coins=COINS):
    return {c: pd.read_pickle(f'{OUT}/ev/{p}_{m}_{c}.pkl') for c in coins}


def feat_list(E, drop_blocks=()):
    skip = {'coin', 'i', 't', 'end', 'lab_end', 'lab_end_t', 'ret'}
    cols = [x for x in E.columns if x not in skip]
    return [x for x in cols if not any(x.startswith(b) for b in drop_blocks)]


# ------------------------------------------------------------------ leak test
def leak():
    global TRUNC
    rng = np.random.default_rng(0)
    full = {c: load(c) for c in ('BTC', 'SOL', 'DOGE')}
    total = 0
    for cut in ('2022-05-11 13:00', '2023-11-02 07:00', '2025-03-19 22:00'):
        TRUNC = pd.Timestamp(cut, tz=TZ)
        mk = market1h()
        for c, F in full.items():
            _, T = build_one((c, mk))
            idx = T.index[-400:]
            a, b = F.loc[idx, T.columns].values.astype(float), T.loc[idx].values.astype(float)
            bad = ~(np.isclose(a, b, rtol=1e-4, atol=1e-6) | (np.isnan(a) & np.isnan(b)))
            cols = T.columns[bad.any(axis=0)]
            n = int(bad.sum())
            # event features on the truncated frame vs full frame, for two primaries
            for p, m in ((10, 3), (48, 4)):
                sf, st_ = st_state(F, p, m), st_state(T, p, m)
                u1, e1 = event_features(F, sf, p)
                u2, e2 = event_features(T, st_, p)
                e1 = e1[(e1.index >= len(T) - 400) & (e1.index < len(T))]
                e2 = e2[e2.index >= len(T) - 400]
                if not e1.index.equals(e2.index):
                    n += 1
                    print('  event index mismatch', c, p, m)
                else:
                    x, y = e1.values.astype(float), e2.values.astype(float)
                    n += int((~(np.isclose(x, y, rtol=1e-6) | (np.isnan(x) & np.isnan(y)))).sum())
            total += n
            print(cut, c, 'mismatches', n, list(cols[:10]), flush=True)
    TRUNC = None
    print('LEAK TEST TOTAL MISMATCHES:', total)


# ------------------------------------------------------------------ labels at a refit date
_ORC = {}


def oracle_at(c, f, cut):
    key = (c, str(cut))
    if key not in _ORC:
        past = f.close.values[f.index < cut]
        if len(past) < 800:
            _ORC[key] = (np.zeros(0), 0)
        else:
            _ORC[key] = oracle_labels(past, 1.0, return_final=True)
    return _ORC[key]


def train_set(EV, closes, cut, label):
    """events whose label is known EMB bars before cut."""
    lim = (cut - EMB * H1).value
    Xs, ys = [], []
    for c, E in EV.items():
        ok = (E.lab_end_t.values <= lim)
        if label == 'oracle':
            lab, fin = oracle_at(c, closes[c], cut)
            ok &= E.lab_end.values < fin - EMB
            e = E[ok]
            y = np.array([np.mean(lab[a:b] == 1) > 0.5 for a, b in zip(e.i.values, e.lab_end.values)], dtype=int)
        else:
            e = E[ok]
            y = (e.ret.values - RT_COST > 0).astype(int)
        Xs.append(e)
        ys.append(y)
    return pd.concat(Xs), np.concatenate(ys)


def lgbp(p):
    return dict(objective='binary', learning_rate=p['lr'], num_leaves=p['leaves'], min_data_in_leaf=p['min_leaf'],
                feature_fraction=p['ff'], bagging_fraction=p['bf'], bagging_freq=1, lambda_l2=p['l2'], verbose=-1,
                num_threads=4, seed=0, deterministic=True)


def fit(EV, closes, cut, p, label, drop=()):
    import lightgbm as lgb
    E, y = train_set(EV, closes, cut, label)
    cols = feat_list(E, drop)
    m = lgb.train(lgbp(p), lgb.Dataset(E[cols].values.astype(np.float32), y, feature_name=cols), num_boost_round=p['trees'])
    return m, cols, len(y), float(y.mean())


# ------------------------------------------------------------------ states and metrics
def meta_state(st, ups, acc):
    """ON only inside accepted up-segments."""
    n = len(st)
    a = np.zeros(n, bool)
    lastup = np.full(n, -1)
    lastup[ups] = np.arange(len(ups))
    lastup = np.maximum.accumulate(lastup)
    ok = lastup >= 0
    a[ok] = acc[lastup[ok]]
    return (st.astype(bool) & a).astype(np.int8)


def metrics(state, tr, lc):
    fpt, mm, md, mg, n = lag_metrics(np.asarray(state, np.int64), np.asarray(tr, np.int64), np.asarray(lc, float))
    return dict(mcc=matthews_corrcoef(tr, state), fpt=fpt, missed=mm, delay=md, giveback=mg, n_seg=n)


def truth(f, k=1.0):
    return (oracle_labels(f.close.values, k) == 1).astype(int)


def objective(rows):
    R = pd.DataFrame(rows)
    return float(R.mcc.mean() - 0.1 * max(0.0, R.fpt.median() - 2.0)), float(R.mcc.mean()), float(R.fpt.median())


def window(idx, lo, hi, trim=0):
    w = np.flatnonzero((idx >= lo) & (idx < hi))
    return w[:-trim] if trim else w


def yr(Y):
    return pd.Timestamp(f'{Y}-01-01', tz=TZ)


# ------------------------------------------------------------------ tuning (dev folds only)
def tune():
    import optuna
    closes = {c: load(c)[['open', 'high', 'low', 'close']] for c in COINS}
    T = {c: truth(closes[c]) for c in EVAL}
    # plain SuperTrend benchmark on dev
    ind = []
    stc = {}
    for p, m in ST_GRID:
        rows = []
        for c in EVAL:
            f = closes[c]
            st = st_state(f, p, m)
            stc[(p, m, c)] = st
            for Y in FOLDS:
                w = window(f.index, yr(Y), yr(Y + 1))
                if len(w) > 1000:
                    rows.append(metrics(st[w], T[c][w], np.log(f.close.values[w])))
        o, mc, fp = objective(rows)
        ind.append(dict(p=p, m=m, obj=o, mcc=mc, fpt_med=fp))
    I = pd.DataFrame(ind).sort_values('obj', ascending=False)
    I.to_csv('results_stage26_dev_supertrend.csv', index=False)
    print(I.head(8).to_string(), flush=True)
    EVc = {pm: load_events(*pm) for pm in PRIMS}
    log = []

    def obj(trial):
        pm = PRIMS[trial.suggest_int('prim', 0, len(PRIMS) - 1)]
        label = trial.suggest_categorical('label', ['ret', 'oracle'])
        p = dict(leaves=trial.suggest_int('leaves', 4, 31, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 500, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 20, 500, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        EV = EVc[pm]
        pr = {}
        for Y in FOLDS:
            mdl, cols, ntr, base = fit(EV, closes, yr(Y), p, label)
            for c in EVAL:
                E = EV[c]
                sel = (E.t >= yr(Y) - pd.Timedelta('60D')) & (E.t < yr(Y + 1))
                pr[(Y, c)] = (E.i.values, np.where(sel.values, np.nan, 1.0), sel.values)
                pv = pr[(Y, c)][1]
                pv[sel.values] = mdl.predict(E.loc[sel, cols].values.astype(np.float32))
        best = (-np.inf, None, None, None)
        for tau in TAUS:
            rows = []
            for (Y, c), (ups, pv, _) in pr.items():
                f = closes[c]
                st = stc[(pm[0], pm[1], c)]
                ms = meta_state(st, ups, pv >= tau)
                w = window(f.index, yr(Y), yr(Y + 1))
                if len(w) > 1000:
                    rows.append(metrics(ms[w], T[c][w], np.log(f.close.values[w])))
            o, mc, fp = objective(rows)
            if o > best[0]:
                best = (o, tau, mc, fp)
        trial.set_user_attr('tau', best[1])
        log.append(dict(p, prim=str(pm), label=label, obj=best[0], tau=best[1], mcc=best[2], fpt_med=best[3]))
        print(trial.number, pm, label, round(best[0], 4), 'tau', best[1], 'mcc', round(best[2], 4), 'fpt', round(best[3], 2), flush=True)
        return best[0]

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(obj, n_trials=40)
    bp = dict(study.best_params)
    prim = PRIMS[bp.pop('prim')]
    label = bp.pop('label')
    b = I.iloc[0]
    json.dump(dict(params=bp, prim=prim, label=label, tau=study.best_trial.user_attrs['tau'], dev_obj=study.best_value,
                   best_supertrend=dict(p=int(b.p), m=float(b.m), obj=float(b.obj), mcc=float(b.mcc), fpt_med=float(b.fpt_med)),
                   trials=log, frozen_at=str(pd.Timestamp.utcnow())), open('FROZEN_stage26.json', 'w'), indent=1, default=str)
    print('FROZEN', prim, label, bp, 'tau', study.best_trial.user_attrs['tau'], 'obj', study.best_value,
          '| BEST ST', dict(b), flush=True)


# ------------------------------------------------------------------ test (run once)
def segs_net(state, c, lo_i, hi_i):
    """net return of every ON segment that starts in [lo_i, hi_i)."""
    fl = np.r_[state[0], np.diff(state.astype(int))]
    st = np.flatnonzero(fl == 1)
    st = st[(st >= lo_i) & (st < hi_i)]
    dn = np.flatnonzero(fl == -1)
    k = np.searchsorted(dn, st, side='right')
    en = np.where(k < len(dn), dn[np.minimum(k, len(dn) - 1)], len(c) - 1)
    return c[en] / c[st] - 1 - RT_COST


def portfolio(states, closes, lo, hi):
    pnl = []
    for c, s in states.items():
        f = closes[c]
        r = f.close.pct_change().shift(-1).fillna(0).values          # return earned from t to t+1
        pos = s.astype(float)
        tc = COST * np.abs(np.diff(pos, prepend=0))
        x = pd.Series(pos * r - tc, index=f.index)
        pnl.append(x[(x.index >= lo) & (x.index < hi)])
    P = pd.concat(pnl, axis=1).fillna(0).sum(axis=1) / len(states)
    eq = (1 + P).cumprod()
    yrs = len(P) / 8760
    return dict(sharpe=float(P.mean() / P.std() * np.sqrt(8760)), cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                maxdd=float((eq / eq.cummax() - 1).min())), P


def test():
    fz = json.load(open('FROZEN_stage26.json'))
    p, prim, label, tau = fz['params'], tuple(fz['prim']), fz['label'], fz['tau']
    bs = fz['best_supertrend']
    closes = {c: load(c)[['open', 'high', 'low', 'close']] for c in COINS}
    EV = load_events(*prim)
    variants = {'META': (), 'META ablation: no 4h flow/deriv/cross': ('h4_flow_', 'h4_deriv_', 'h4_cross_', 'h1flow_')}
    acc = {v: {c: np.ones(len(EV[c]), bool) for c in EVAL} for v in variants}
    prob = {c: np.full(len(EV[c]), np.nan) for c in EVAL}
    info = []
    for v, drop in variants.items():
        for Y in range(T0.year, T1.year + 1):
            lo, hi = yr(Y), min(yr(Y + 1), T1)
            mdl, cols, ntr, base = fit(EV, closes, lo, p, label, drop)
            info.append(dict(variant=v, year=Y, n_train=ntr, base_rate=base))
            for c in EVAL:
                E = EV[c]
                sel = ((E.t >= lo - (pd.Timedelta('60D') if Y == T0.year else pd.Timedelta(0))) & (E.t < hi)).values
                pv = mdl.predict(E.loc[sel, cols].values.astype(np.float32))
                acc[v][c][sel] = pv >= tau
                if v == 'META':
                    prob[c][sel] = pv
        print('fitted', v, flush=True)
    print(pd.DataFrame(info).to_string(), flush=True)
    rows, trades, S_all = [], [], {}
    for k in (1.0, 2.0):
        T = {c: truth(closes[c], k) for c in EVAL}
        for c in EVAL:
            f = closes[c]
            g = s25.load(c)
            j = align4h(g, f.index)
            s4 = s25.indicator_state(g, 'SuperTrend', dict(p=48, m=4))
            cands = {'META': meta_state(st_state(f, *prim), EV[c].i.values, acc['META'][c]),
                     'META ablation: no 4h flow/deriv/cross': meta_state(st_state(f, *prim), EV[c].i.values,
                                                                        acc['META ablation: no 4h flow/deriv/cross'][c]),
                     f'primary unfiltered ST{prim}': st_state(f, *prim),
                     f"BEST plain ST({bs['p']},{bs['m']})": st_state(f, bs['p'], bs['m']),
                     'ST(10,3) 1h': st_state(f, 10, 3),
                     'ST(48,4) 4h projected': np.where(j >= 0, s4[np.maximum(j, 0)], 0).astype(np.int8)}
            w = window(f.index, T0, T1, trim=TRIM)
            for name, s in cands.items():
                rows.append(dict(truth_k=k, strategy=name, coin=c, **metrics(s[w], T[c][w], np.log(f.close.values[w]))))
                if k == 1.0:
                    S_all.setdefault(name, {})[c] = s
                    net = segs_net(s, f.close.values, w[0], w[-1] + 1)
                    trades.append(dict(strategy=name, coin=c, n=len(net), wins=int((net > 0).sum()), sum_net=float(net.sum())))
    R = pd.DataFrame(rows)
    R.to_csv('results_stage26_per_coin.csv', index=False)
    Tr = pd.DataFrame(trades)
    pf = []
    for name, st in S_all.items():
        d, _ = portfolio(st, closes, T0, T1)
        t = Tr[Tr.strategy == name]
        pf.append(dict(strategy=name, precision=t.wins.sum() / max(t.n.sum(), 1), n_trades=int(t.n.sum()),
                       mean_net=t.sum_net.sum() / max(t.n.sum(), 1),
                       exposure=float(np.mean([s[(closes[c].index >= T0) & (closes[c].index < T1)].mean() for c, s in st.items()])), **d))
    PF = pd.DataFrame(pf).set_index('strategy')
    S = R[R.truth_k == 1.0].groupby('strategy').agg(mcc_med=('mcc', 'median'), mcc_mean=('mcc', 'mean'), fpt_med=('fpt', 'median'),
                                                    delay_med=('delay', 'median'), missed_med=('missed', 'median'),
                                                    giveback_med=('giveback', 'median'))
    S2 = R[R.truth_k == 2.0].groupby('strategy').mcc.median().rename('mcc_med_k2')
    S = S.join(S2).join(PF).sort_values('mcc_med', ascending=False)
    S.to_csv('results_stage26_summary.csv')
    pd.set_option('display.width', 250)
    print(S.round(3).to_string())
    # per year
    yrows = []
    for Y in range(T0.year, T1.year + 1):
        T = {c: truth(closes[c]) for c in EVAL} if Y == T0.year else T
        for name in ('META', f"BEST plain ST({bs['p']},{bs['m']})", f'primary unfiltered ST{prim}'):
            ms = []
            for c in EVAL:
                f = closes[c]
                w = window(f.index, yr(Y), min(yr(Y + 1), T1), trim=TRIM if Y == T1.year else 0)
                ms.append(matthews_corrcoef(T[c][w], S_all[name][c][w]))
            yrows.append(dict(year=Y, strategy=name, mcc_med=float(np.median(ms))))
    Yd = pd.DataFrame(yrows).pivot(index='year', columns='strategy', values='mcc_med')
    Yd.to_csv('results_stage26_years.csv')
    print(Yd.round(3).to_string())
    bname = f"BEST plain ST({bs['p']},{bs['m']})"
    Rk = R[R.truth_k == 1.0]
    M, B = Rk[Rk.strategy == 'META'].set_index('coin'), Rk[Rk.strategy == bname].set_index('coin')
    d_mcc = M.mcc.median() - B.mcc.median()
    wins = int((M.mcc > B.mcc).sum())
    p1 = d_mcc >= 0.02 and wins >= 7
    p2 = PF.loc['META', 'precision'] >= PF.loc[bname, 'precision'] + 0.05
    p3 = PF.loc['META', 'sharpe'] >= PF.loc[bname, 'sharpe']
    print(f'primary {prim} label {label} tau {tau} | BEST {bname}')
    print(f"1 accuracy: dMCC {d_mcc:+.3f}, wins {wins}/10 -> {p1}")
    print(f"2 precision: {PF.loc['META', 'precision']:.3f} vs {PF.loc[bname, 'precision']:.3f} -> {p2}")
    print(f"3 Sharpe: {PF.loc['META', 'sharpe']:.2f} vs {PF.loc[bname, 'sharpe']:.2f} -> {p3}")
    print('PRIMARY:', bool(p1 and p2 and p3))
    pd.to_pickle(dict(S_all=S_all, prob=prob, acc=acc), f'{OUT}/test_states.pkl')


if __name__ == '__main__':
    dict(build=build, leak=leak, events=events, tune=tune, test=test)[sys.argv[1]]()

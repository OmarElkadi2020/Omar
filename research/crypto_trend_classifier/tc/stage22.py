"""Stage 22: meta trend filter (see PREREG_stage22_meta_trend_filter.md).
python -m tc.stage22 build crypto|india ; tune crypto|india ; test crypto|india"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .stage16 import load_crypto, load_india, rsi, er, chop, r2, run_book, alpha, stats, CD, TZ, EMB
from .stage20 import indicators, donchian_state
from .evaluate import supertrend

H = 30
BENCH = ['EMA50', 'EMA200', 'GOLDEN', 'MACD', 'TSMOM30', 'TSMOM90', 'DONCHIAN20_10', 'SUPERTREND10_3', 'ADX25_DI']
MAJORS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT', 'TRXUSDT']
CFG = dict(
    crypto=dict(cost=0.001, bpy=365, week='W-SUN', folds=(2019, 2020, 2021), test=('2022-01-01', '2026-09-01'),
                step=1),
    india=dict(cost=0.0015, bpy=252, week='W-FRI', folds=(2006, 2007, 2008, 2009, 2010),
               test=('2011-01-01', '2026-09-25'), step=3))


def st_frame(C, H_, L, n, m):
    S = {}
    need = max(60, 3 * n)
    for s in C.columns:
        f = pd.DataFrame(dict(high=H_[s], low=L[s], close=C[s])).dropna()
        if len(f) < n + 2:
            continue
        v = pd.Series(supertrend(f, n, m).astype(np.float32), index=f.index)
        # causal availability: hide values until `need` bars of history exist AT THAT DATE
        # (a total-length check would leak whether the asset survives into the future)
        S[s] = v.where(np.arange(len(v)) >= need)
    return pd.DataFrame(S).reindex(index=C.index, columns=C.columns)


def ts_features(C, H_, L, add):
    """add(name, wide_frame) is called per feature (keeps memory flat)."""
    lc = np.log(C)
    r = lc.diff().clip(-1, 1)
    s30 = r.rolling(30, min_periods=20).std()
    for n in (20, 50, 100, 200):
        e = C.ewm(span=n, adjust=False).mean()
        add(f'ema_d_{n}', (lc - np.log(e)) / s30)
        add(f'ema_slope_{n}', (np.log(e) - np.log(e.shift(5))) / s30)
    add('golden', (C.rolling(50).mean() > C.rolling(200).mean()).astype(float).where(C.rolling(200).count() == 200))
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    add('macd_hist', (macd - sig) / (C * s30))
    add('macd_up', (macd > sig).astype(float))
    for n in (10, 30, 60, 90, 180, 365):
        add(f'tsmom_{n}', (lc - lc.shift(n)) / (s30 * np.sqrt(n)))
    for n in (20, 55, 100):
        hi, lo = H_.rolling(n).max(), L.rolling(n).min()
        add(f'don_pos_{n}', (C - lo) / (hi - lo).replace(0, np.nan))
    add('don_20_10', donchian_state(C, H_, L, 20, 10).astype(float))
    add('don_55_20', donchian_state(C, H_, L, 55, 20).astype(float))
    for n, m in ((10, 3.0), (20, 4.0)):
        add(f'st_{n}_{int(m)}', st_frame(C, H_, L, n, m))
    up, dn = H_.diff(), -L.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = np.maximum(H_ - L, np.maximum((H_ - C.shift()).abs(), (L - C.shift()).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    ndi = ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    add('adx14', dx.ewm(alpha=1 / 14, adjust=False).mean())
    add('di_diff', (pdi - ndi) / (pdi + ndi).replace(0, np.nan))
    add('rsi14', rsi(C))
    add('er30', er(C, 30))
    add('er90', er(C, 90))
    add('chop14', chop(H_, L, C, 14))
    add('r2_60', r2(lc, 60))
    add('r2_180', r2(lc, 180))
    add('vol_ratio', s30 / r.rolling(365, min_periods=120).std())
    add('dd_365', lc - np.log(H_.rolling(365, min_periods=60).max()))
    for n in (50, 200):
        side = np.sign(C - C.ewm(span=n, adjust=False).mean())
        chg = (side != side.shift()).astype(int)
        add(f'since_x_{n}', np.log1p(_since(chg)))


def _since(chg):
    v = chg.values
    out = np.zeros(v.shape, np.float32)
    cnt = np.zeros(v.shape[1], np.float32)
    for i in range(len(v)):
        cnt = np.where(v[i] > 0, 0, cnt + 1)
        out[i] = cnt
    return pd.DataFrame(out, index=chg.index, columns=chg.columns)


def weekly_features(C, H_, L, week, add):
    W = dict(C=C.resample(week).last(), H=H_.resample(week).max(), L=L.resample(week).min())
    lc = np.log(W['C'])
    rw = lc.diff()
    s = rw.rolling(26, min_periods=10).std()
    e10, e40 = W['C'].ewm(span=10, adjust=False).mean(), W['C'].ewm(span=40, adjust=False).mean()
    F = {'w_ema_d_10': (lc - np.log(e10)) / s, 'w_ema_d_40': (lc - np.log(e40)) / s,
         'w_ema_10_40': (e10 > e40).astype(float).where(W['C'].notna()), 'w_rsi14': rsi(W['C']),
         'w_ret_13': lc - lc.shift(13), 'w_ret_52': lc - lc.shift(52), 'w_st_10_3': st_frame(W['C'], W['H'], W['L'], 10, 3.0)}
    for k, v in F.items():
        add(k, v.reindex(C.index, method='ffill'))


def build(market):
    cfg = CFG[market]
    if market == 'crypto':
        P, _ = load_crypto()
        D20 = pd.read_pickle(f'{CD}/crypto20_P.pkl')
        elig = D20['elig'].reindex(index=P['C'].index, columns=P['C'].columns).fillna(False)
    else:
        Di = pd.read_pickle(f'{CD}/india_P.pkl')
        P = load_india()
        elig = Di['elig']
        P = {k: v.reindex(index=elig.index, columns=elig.columns) for k, v in P.items()}
    C, H_, L = P['C'].astype(float), P['H'].astype(float), P['L'].astype(float)
    cols = []

    def add(name, v):
        cols.append(v.where(elig).stack().dropna().rename(name).astype(np.float32))

    ts_features(C, H_, L, add)
    weekly_features(C, H_, L, cfg['week'], add)
    ind = indicators(P)
    vote = sum(ind[k].astype(int) for k in BENCH)
    add('n_ind_on', vote.astype(float))
    lex = np.log(P['EX'].astype(float))
    lr = np.log(C).diff().clip(-1, 1)
    s30 = lr.rolling(30, min_periods=20).std()
    add('y', ((lex.shift(-H) - lex) / (s30 * np.sqrt(H))).clip(-4, 4))
    X = pd.concat(cols, axis=1)
    X.index.names = ['date', 'asset']
    # market context: proxy = BTC (crypto) or EW universe index (India), same trend features, raw
    if market == 'crypto':
        px = C['BTCUSDT']
    else:
        px = np.exp(lr.where(elig).mean(axis=1).fillna(0).cumsum())
    pc = pd.DataFrame({'m': px})
    ctx = {}
    lp = np.log(pc.m)
    rp = lp.diff()
    sp = rp.rolling(30, min_periods=20).std()
    for n in (50, 200):
        ctx[f'mkt_ema_d_{n}'] = (lp - np.log(pc.m.ewm(span=n, adjust=False).mean())) / sp
    for n in (30, 90, 180):
        ctx[f'mkt_tsmom_{n}'] = (lp - lp.shift(n)) / (sp * np.sqrt(n))
    ctx['mkt_vol_ratio'] = sp / rp.rolling(365, min_periods=120).std()
    ctx['breadth_ema50'] = ((C > C.ewm(span=50, adjust=False).mean()) & elig).sum(axis=1) / elig.sum(axis=1)
    ctx['breadth_ema200'] = ((C > C.ewm(span=200, adjust=False).mean()) & elig).sum(axis=1) / elig.sum(axis=1)
    X = X.join(pd.DataFrame(ctx).astype(np.float32), on='date')
    X.to_pickle(f'{CD}/s22_{market}_X.pkl')
    ind['VOTE'] = vote >= 5
    rex = (lex.shift(-1) - lex).clip(-1, 1)
    dvm = P['DV'].rolling(30, min_periods=10).median().where(elig)
    pd.to_pickle(dict(rex=rex.astype(np.float32), elig=elig, C=C.astype(np.float32), ind=ind,
                      dvrank=dvm.rank(axis=1, ascending=False).astype(np.float32)), f'{CD}/s22_{market}_P.pkl')
    print(market, X.shape, X.y.notna().sum(), flush=True)


# ------------------------------------------------------------------ evaluation
def load(market):
    X = pd.read_pickle(f'{CD}/s22_{market}_X.pkl')
    D = pd.read_pickle(f'{CD}/s22_{market}_P.pkl')
    D['rex'] = np.expm1(D['rex'].astype(float))
    return X, D


def slots(market, D, lo, hi):
    rex = D['rex'].loc[lo:hi]
    if market == 'crypto':
        live = rex[MAJORS].notna() & D['C'].loc[lo:hi, MAJORS].notna()
        return rex[MAJORS], live.astype(float) * 0.1
    top = (D['dvrank'].loc[lo:hi] <= 100) & D['elig'].loc[lo:hi] & rex.notna()
    return rex, top.astype(float) * 0.01


def port(mask, rex, slot, cost):
    w = mask.reindex(index=rex.index, columns=rex.columns).fillna(False).astype(float) * slot
    return run_book(w, rex, cost, 1)[0]


def bench(market, D, lo, hi):
    rex, slot = slots(market, D, lo, hi)
    cost = CFG[market]['cost']
    R = {k: port(D['ind'][k].loc[lo:hi], rex, slot, cost) for k in BENCH + ['VOTE']}
    R['BUY_HOLD'] = port(pd.DataFrame(True, index=rex.index, columns=rex.columns), rex, slot, cost)
    return pd.DataFrame(R)


def state(P, span, din, dout):
    s = P.ewm(span=span, adjust=False, ignore_na=True).mean() if span > 1 else P
    v = s.values
    out = np.zeros(v.shape, bool)
    cur = np.zeros(v.shape[1], bool)
    for i in range(len(v)):
        x = v[i]
        cur = np.where(np.isnan(x), cur, np.where(cur, x >= dout, x > din))
        out[i] = cur
    return pd.DataFrame(out, index=P.index, columns=P.columns)


def lgbp(p):
    return dict(objective='huber', alpha=1.0, learning_rate=p['lr'], num_leaves=p['leaves'], min_data_in_leaf=p['min_leaf'],
                feature_fraction=p['ff'], bagging_fraction=p['bf'], bagging_freq=1, lambda_l2=p['l2'], verbose=-1,
                num_threads=4, seed=0, deterministic=True)


def fit(X, cut, p, step):
    d = X.index.get_level_values(0)
    ok = (d + pd.Timedelta(days=int(H * 1.5) + 2) < cut - EMB) & X.y.notna().values
    t = X[ok]
    if step > 1:
        dd = t.index.get_level_values(0)
        t = t[((dd - dd.min()).days % step) == 0]
    cols = [c for c in X.columns if c != 'y']
    return lgb.train(lgbp(p), lgb.Dataset(t[cols].values, t.y.values, feature_name=cols), num_boost_round=p['trees']), cols


def predict_year(X, m, cols, lo, hi, assets=None):
    d = X.index.get_level_values(0)
    w = (d >= lo - pd.Timedelta('60D')) & (d < hi)          # 60-day warm-up for the smoothing only
    if assets is not None:
        w &= X.index.get_level_values(1).isin(assets)
    return pd.Series(m.predict(X.loc[w, cols].values), index=X.index[w]).unstack()


def filter_returns(market, X, D, preds, p, lo, hi):
    rex, slot = slots(market, D, lo, hi)
    st = state(preds.reindex(columns=rex.columns), p['span'], p['din'], p['dout']).loc[lo:hi]
    return port(st, rex, slot, CFG[market]['cost']), st


def tune(market):
    import optuna
    cfg = CFG[market]
    X, D = load(market)
    assets = MAJORS if market == 'crypto' else None
    B = {Y: bench(market, D, pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)) for Y in cfg['folds']}
    log = []

    def obj(trial):
        p = dict(leaves=trial.suggest_int('leaves', 7, 63, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 500, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 200, 5000, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        preds = {}
        for Y in cfg['folds']:
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
            m, cols = fit(X, lo, p, cfg['step'])
            preds[Y] = predict_year(X, m, cols, lo, hi, assets)
        best = (-np.inf, None)
        for span in (1, 5, 10, 20):
            for din in (-0.1, 0.0, 0.1):
                for dout in (-0.2, -0.1, 0.0):
                    if dout > din:
                        continue
                    q = dict(span=span, din=din, dout=dout)
                    ts = []
                    for Y in cfg['folds']:
                        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
                        r, _ = filter_returns(market, X, D, preds[Y], q, lo, hi)
                        a, t, _ = alpha(r, B[Y].loc[r.index])
                        ts.append(t if np.isfinite(t) else 0.0)
                    v = float(np.mean(ts))
                    if v > best[0]:
                        best = (v, dict(q, t_folds=ts))
        trial.set_user_attr('filter', best[1])
        log.append(dict(p, t=best[0], **best[1]))
        print(trial.number, round(best[0], 2), best[1], flush=True)
        return best[0]

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=25)
    f = st.best_trial.user_attrs['filter']
    params = dict(st.best_params, span=f['span'], din=f['din'], dout=f['dout'])
    json.dump(dict(params=params, dev_t=st.best_value, trials=log, frozen_at=str(pd.Timestamp.utcnow())),
              open(f'FROZEN_stage22_{market}.json', 'w'), indent=1, default=str)
    print('BEST', params, st.best_value, flush=True)


def quality(mask, D, market, lo, hi):
    rex, slot = slots(market, D, lo, hi)
    C = D['C'].astype(float).loc[:, rex.columns]
    lc = np.log(C)
    f30 = (lc.shift(-30) - lc).loc[rex.index]
    m = mask.reindex(index=rex.index, columns=rex.columns).fillna(False).astype(bool)
    L = (slot > 0) & f30.notna()
    on, off = m & L, ~m & L
    crash, rally = (f30 < np.log(0.8)) & L, (f30 > np.log(1.2)) & L
    flips = (m.astype(int).diff().abs() > 0) & (slot > 0)
    return dict(fwd30_on=float(np.expm1(f30[on].stack()).mean()), fwd30_off=float(np.expm1(f30[off].stack()).mean()),
                crashes_avoided=float((off & crash).values.sum() / max(crash.values.sum(), 1)),
                rallies_captured=float((on & rally).values.sum() / max(rally.values.sum(), 1)),
                flips_per_asset_year=float(flips.values.sum() / max((slot > 0).values.sum(), 1) * CFG[market]['bpy']),
                time_in_market=float(on.values.sum() / max(L.values.sum(), 1)))


def test(market):
    cfg = CFG[market]
    p = json.load(open(f'FROZEN_stage22_{market}.json'))['params']
    X, D = load(market)
    t0, t1 = (pd.Timestamp(x, tz=TZ) for x in cfg['test'])
    assets = MAJORS if market == 'crypto' else None
    parts = []
    for Y in range(t0.year, t1.year + 1):
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), min(pd.Timestamp(f'{Y + 1}-01-01', tz=TZ), t1)
        m, cols = fit(X, lo, p, cfg['step'])
        pr = predict_year(X, m, cols, lo, hi, assets)
        parts.append(pr if Y == t0.year else pr.loc[lo:])
        print(Y, flush=True)
    preds = pd.concat(parts)
    preds = preds[~preds.index.duplicated(keep='last')]
    preds.to_pickle(f'{CD}/s22_{market}_preds.pkl')
    r, st = filter_returns(market, X, D, preds, p, t0, t1)
    B = bench(market, D, t0, t1).loc[r.index]
    a, t, mm = alpha(r, B)
    rows = [dict(filter='MODEL', alpha_ann=a * cfg['bpy'], alpha_t=t, **stats(r, cfg['bpy']), **quality(st, D, market, t0, t1))]
    for k in B:
        mk = D['ind'][k].loc[t0:t1] if k != 'BUY_HOLD' else pd.DataFrame(True, index=st.index, columns=st.columns)
        rows.append(dict(filter=k, **stats(B[k], cfg['bpy']), **quality(mk, D, market, t0, t1)))
    T = pd.DataFrame(rows)
    Cc = pd.concat([r.rename('MODEL'), B], axis=1)
    yr = Cc.groupby(Cc.index.year).apply(lambda g: np.expm1(np.log1p(g).sum()))
    T.to_csv(f'results_stage22_{market}.csv', index=False)
    yr.to_csv(f'results_stage22_{market}_years.csv')
    Cc.to_pickle(f'{CD}/s22_{market}_returns.pkl')
    pd.set_option('display.width', 260)
    print(T.round(3).to_string())
    print(yr.round(2).to_string())
    ok1, ok2 = a > 0 and t >= 2.5, T.sharpe.iloc[0] > T.sharpe.iloc[1:].max()
    print(f'{market}: P alpha t>=2.5: {ok1}  Sharpe > every benchmark: {ok2}  PASS: {bool(ok1 and ok2)}')


if __name__ == '__main__':
    dict(build=build, tune=tune, test=test)[sys.argv[1]](sys.argv[2])

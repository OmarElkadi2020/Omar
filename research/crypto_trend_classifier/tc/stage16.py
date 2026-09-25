"""Stage 16: cross-sectional trend model (see PREREG_stage16_cross_sectional_trend.md).
python -m tc.stage16 build crypto|stocks ; tune crypto|stocks ; test crypto|stocks"""
import glob
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm

CD = 'cache_stage16'
EXCL = re.compile(r'(UP|DOWN|BULL|BEAR)USDT$')
EXCL_NAMES = {'WBTCUSDT', 'WBETHUSDT', 'BETHUSDT'}
EMB = pd.Timedelta('7D')
CFG = dict(
    crypto=dict(cost=0.001, top=100, min_hist=60, train_end='2021-01-01', val=('2021-01-01', '2022-01-01'),
                test=('2022-01-01', '2026-09-01'), years=range(2022, 2027), bpy=365),
    stocks=dict(cost=0.0005, top=None, min_hist=252, train_end='2006-01-01', val=('2006-01-01', '2011-01-01'),
                test=('2011-01-01', '2026-09-23'), years=range(2011, 2027), bpy=252))
HS = (3, 7, 14, 30)
TZ = 'UTC'


# ------------------------------------------------------------------ panels
def load_crypto():
    O, H, L, C, V, C4, H4, L4 = {}, {}, {}, {}, {}, {}, {}, {}
    for f in sorted(glob.glob('bnc/k4h/*.pkl')):
        s = os.path.basename(f)[:-4]
        if EXCL.search(s) or s in EXCL_NAMES:
            continue
        d = pd.read_pickle(f)
        d = d[(d.close > 0) & (d.volume > 0)]
        if len(d) < 6 * 90:
            continue
        g = d.resample('1D').agg(dict(open='first', high='max', low='min', close='last', volume='sum')).dropna()
        O[s], H[s], L[s], C[s], V[s] = g.open, g.high, g.low, g.close, g.volume * g.close
        C4[s], H4[s], L4[s] = d.close, d.high, d.low
    P = {k: pd.DataFrame(v).sort_index() for k, v in dict(O=O, H=H, L=L, C=C, DV=V).items()}
    P4 = {k: pd.DataFrame(v).sort_index() for k, v in dict(C=C4, H=H4, L=L4).items()}
    # execution price for signal date D: close of the 4h bar [D+1 00:00, D+1 04:00)
    ex = P4['C'][P4['C'].index.hour == 0]
    ex.index = ex.index.normalize() - pd.Timedelta('1D')
    P['EX'] = ex.reindex(P['C'].index)
    return P, P4


def load_stocks():
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                                                            'adj_close'])
    d['date'] = pd.to_datetime(d.date).dt.tz_localize(TZ)
    d = d.drop_duplicates(['date', 'ticker'])
    have = {x[:-4] for x in os.listdir('cache_stage12')}
    d = d[d.ticker.isin(have)]
    P = {k: d.pivot(index='date', columns='ticker', values=v).sort_index()
         for k, v in dict(O='open', H='high', L='low', C='close', V='volume', A='adj_close').items()}
    for k in ('O', 'H', 'L', 'C', 'A'):
        P[k] = P[k].where(P[k] > 0)
    P['DV'] = P['C'] * P['V']
    # execution for signal date D: next open, dividend/split adjusted
    P['EX'] = (P['O'] * P['A'] / P['C']).shift(-1)
    return P


# ------------------------------------------------------------------ features
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def er(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)


def chop(h, l, c, n):
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    rng = h.rolling(n).max() - l.rolling(n).min()
    return np.log10(tr.rolling(n).sum() / rng.replace(0, np.nan)) / np.log10(n)


def r2(lc, n):
    t = pd.Series(np.arange(len(lc), dtype=float), index=lc.index)
    return lc.rolling(n).corr(t) ** 2


def daily_features(P, mkt_ret, lags):
    C, H, L, DV = P['C'], P['H'], P['L'], P['DV']
    lc = np.log(C)
    r = lc.diff().clip(-1, 1)
    s30 = r.rolling(30, min_periods=20).std()
    F = {}
    for k in lags:
        F[f'ret_{k}'] = lc - lc.shift(k)
        F[f'nret_{k}'] = F[f'ret_{k}'] / (s30 * np.sqrt(k))
    for n in (7, 30, 90):
        F[f'vol_{n}'] = r.rolling(n, min_periods=max(5, n // 2)).std()
    F['vol_ratio'] = F['vol_7'] / F['vol_90']
    F['max_30'] = r.rolling(30).max()
    F['min_30'] = r.rolling(30).min()
    F['skew_30'] = r.rolling(30).skew()
    for n in (10, 30, 100):
        F[f'ema_d_{n}'] = (lc - np.log(C.ewm(span=n, adjust=False).mean())) / s30
    F['dd_90'] = lc - np.log(H.rolling(90, min_periods=30).max())
    F['ru_90'] = lc - np.log(L.rolling(90, min_periods=30).min())
    for n in (20, 60):
        hi, lo = H.rolling(n).max(), L.rolling(n).min()
        F[f'don_{n}'] = (C - lo) / (hi - lo).replace(0, np.nan)
    F['er_14'], F['er_60'] = er(C, 14), er(C, 60)
    F['chop_14'] = chop(H, L, C, 14)
    F['r2_30'], F['r2_90'] = r2(lc, 30), r2(lc, 90)
    F['rsi_14'] = rsi(C)
    F['size'] = np.log(DV.rolling(30, min_periods=10).median())
    F['dv_chg'] = np.log(DV.rolling(7).mean() / DV.rolling(30).mean())
    cov = r.rolling(60, min_periods=30).cov(mkt_ret)
    beta = cov.div(mkt_ret.rolling(60, min_periods=30).var(), axis=0)
    F['beta_60'] = beta
    F['resmom_30'] = (r - beta.mul(mkt_ret, axis=0)).rolling(30).sum()
    F['age'] = np.log1p(C.notna().cumsum())
    return F


def mtf_crypto(P4, index):
    C, H, L = P4['C'], P4['H'], P4['L']
    lc = np.log(C)
    r = lc.diff().clip(-1, 1)
    s = r.rolling(42, min_periods=20).std()
    F = {'h4_rsi_14': rsi(C), 'h4_er_42': er(C, 42), 'h4_nret_12': (lc - lc.shift(12)) / (s * np.sqrt(12)),
         'h4_chop_42': chop(H, L, C, 42)}
    hi, lo = H.rolling(42).max(), L.rolling(42).min()
    F['h4_pos_42'] = (C - lo) / (hi - lo).replace(0, np.nan)
    out = {}
    for k, v in F.items():
        v = v[v.index.hour == 20]            # 4h bar 20:00-24:00 closes with the daily bar
        v.index = v.index.normalize()
        out[k] = v.reindex(index)
    return out


def mtf_weekly(P, index):
    W = {k: P[k].resample('W-FRI').agg(a) for k, a in (('C', 'last'), ('H', 'max'), ('L', 'min'))}
    lc = np.log(W['C'])
    F = {'w_ret_4': lc - lc.shift(4), 'w_ret_13': lc - lc.shift(13), 'w_rsi_14': rsi(W['C']), 'w_er_13': er(W['C'], 13)}
    return {k: v.reindex(index, method='ffill') for k, v in F.items()}   # week known at its Friday close


def xrank(df, mask):
    return df.where(mask).rank(axis=1, pct=True) - 0.5


def build(market):
    os.makedirs(CD, exist_ok=True)
    cfg = CFG[market]
    if market == 'crypto':
        P, P4 = load_crypto()
    else:
        P = load_stocks()
    C = P['C']
    r = np.log(C).diff().clip(-1, 1)
    hist = C.notna().cumsum()
    elig = (hist >= cfg['min_hist']) & C.notna() & P['EX'].notna()
    if cfg['top']:
        dvm = P['DV'].rolling(30, min_periods=10).median().where(elig)
        elig = elig & (dvm.rank(axis=1, ascending=False) <= cfg['top'])
    mkt = r.where(elig).mean(axis=1)
    lags = (1, 3, 7, 14, 30, 60, 90, 180) if market == 'crypto' else (1, 3, 7, 14, 30, 60, 90, 180, 252)
    F = daily_features(P, mkt, lags)
    F.update(mtf_crypto(P4, C.index) if market == 'crypto' else mtf_weekly(P, C.index))
    feats = {}
    for k in list(F):
        feats[k] = xrank(F.pop(k), elig).astype(np.float32)
    # raw market context
    lm = mkt.fillna(0).cumsum()
    ctx = pd.DataFrame({f'mkt_ret_{n}': lm - lm.shift(n) for n in (7, 30, 90)})
    lead = 'BTCUSDT' if market == 'crypto' else None
    ctx['lead_ret_30'] = (np.log(C[lead]) - np.log(C[lead].shift(30))) if lead else ctx['mkt_ret_30']
    r30 = np.log(C) - np.log(C.shift(30))
    ctx['breadth_30'] = (r30.where(elig) > 0).sum(axis=1) / elig.sum(axis=1)
    ctx['disp_30'] = r30.where(elig).std(axis=1)
    # execution-based returns and targets
    lex = np.log(P['EX'])
    rex = (lex.shift(-1) - lex).clip(-1, 1)            # return earned by a position formed at signal date D
    T = {f'y{H}': xrank(lex.shift(-H) - lex, elig & (lex.shift(-H) - lex).notna()) for H in HS}
    rows = []
    for k, v in list(feats.items()) + list(T.items()):
        rows.append(v.where(elig).stack().dropna().rename(k).astype(np.float32))   # pandas 3 keeps NaN in stack
    X = pd.concat(rows, axis=1)
    X = X[X.index.get_level_values(1).isin(C.columns)]
    X.index.names = ['date', 'asset']
    X = X.join(ctx.astype(np.float32), on='date')
    X.to_pickle(f'{CD}/{market}_X.pkl')
    pd.to_pickle(dict(rex=rex.astype(np.float32), elig=elig, r=r.astype(np.float32),
                      dv=P['DV'].rolling(30, min_periods=10).median().astype(np.float32),
                      C=C.astype(np.float32), vol60=r.rolling(60, min_periods=30).std().astype(np.float32)),
                 f'{CD}/{market}_P.pkl')
    print(market, X.shape, X.index.get_level_values(0).min(), X.index.get_level_values(0).max(),
          'assets/day', elig.sum(axis=1).median(), flush=True)


# ------------------------------------------------------------------ portfolios, factors, alpha
def book(score, q=0.2, long_only=False):
    rk = score.rank(axis=1, pct=True)
    n = score.notna().sum(axis=1)
    top = rk > 1 - q
    w = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if not long_only:
        bot = rk <= q
        w = w - bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.mul((n >= 10).astype(float), axis=0)


def run_book(w, rex, cost, hold=1):
    if hold > 1:
        w = w.rolling(hold, min_periods=1).mean()
    w = w.reindex(columns=rex.columns, fill_value=0.0).fillna(0.0)
    gross = (w * rex.fillna(0)).sum(axis=1)
    to = w.diff().abs().sum(axis=1)
    return gross - cost * to, to


def factors(D, market):
    rex, elig, C, dv = D['rex'], D['elig'], D['C'], D['dv']
    f = pd.DataFrame(index=rex.index)
    f['MKT'] = rex.where(elig).mean(axis=1)
    lc = np.log(C.astype(float))
    if market == 'crypto':
        f['BTC'] = rex['BTCUSDT']
        sigs = dict(SIZE=-dv, CMOM=lc - lc.shift(21), STREV=-(lc - lc.shift(7)))
    else:
        sigs = dict(SIZE=-dv, UMD=lc.shift(21) - lc.shift(252), STREV=-(lc - lc.shift(21)),
                    LOWVOL=-D['vol60'])
    for k, s in sigs.items():
        f[k], _ = run_book(book(s.where(elig), q=1 / 3), rex, 0.0)
    return f


def alpha(ret, F, lag=20):
    df = pd.concat([ret.rename('y'), F], axis=1).dropna()
    m = sm.OLS(df.y, sm.add_constant(df.drop(columns='y'))).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return float(m.params['const']), float(m.tvalues['const']), m


def stats(x, bpy):
    x = x.dropna()
    eq = np.log1p(x.clip(lower=-0.99)).cumsum()
    return dict(ann_ret=float(x.mean() * bpy), cagr=float(np.expm1(eq.iloc[-1] * bpy / len(x))),
                sharpe=float(x.mean() / x.std() * np.sqrt(bpy)), max_dd=float((np.exp(eq - eq.cummax()) - 1).min()))


# ------------------------------------------------------------------ model
FEAT_EXCL = re.compile(r'^(y\d+|fwd\d+)$')


def feat_cols(X):
    return [c for c in X.columns if not FEAT_EXCL.match(c)]


def params(p):
    return dict(objective='regression', learning_rate=p['lr'], num_leaves=p['leaves'], min_data_in_leaf=p['min_leaf'],
                feature_fraction=p['ff'], bagging_fraction=p['bf'], bagging_freq=1, lambda_l2=p['l2'],
                verbose=-1, num_threads=4, seed=0, deterministic=True)


def train_rows(X, H, cut, step):
    d = X.index.get_level_values(0)
    ok = (d + pd.Timedelta(days=int(H * 1.5) + 2) < cut - EMB) & X[f'y{H}'].notna().values
    t = X[ok]
    if step > 1:
        dd = t.index.get_level_values(0)
        keep = ((dd - dd.min()).days % step) == 0
        t = t[keep]
    return t


def fit(X, H, cut, p, step):
    t = train_rows(X, H, cut, step)
    cols = feat_cols(X)
    ds = lgb.Dataset(t[cols].values, t[f'y{H}'].values, feature_name=cols, free_raw_data=True)
    return lgb.train(params(p), ds, num_boost_round=p['trees']), cols


def predict(m, cols, X, lo, hi):
    d = X.index.get_level_values(0)
    w = (d >= lo) & (d < hi)
    return pd.Series(m.predict(X.loc[w, cols].values), index=X.index[w])


def load(market):
    X = pd.read_pickle(f'{CD}/{market}_X.pkl')
    D = pd.read_pickle(f'{CD}/{market}_P.pkl')
    # portfolio accounting must use SIMPLE returns: summing w*log-returns hands shorts a fake +sigma^2/2 per bar
    D['rex'] = np.expm1(D['rex'].astype(float))
    return X, D


def evaluate_scores(S, D, market, hold, lo, hi):
    cfg = CFG[market]
    S = S.unstack()
    rex = D['rex'].loc[lo:hi]
    S = S.reindex(index=rex.index)
    ls, to = run_book(book(S), rex, cfg['cost'], hold)
    lo_, to_l = run_book(book(S, long_only=True), rex, cfg['cost'], hold)
    if '_F' not in D:
        D['_F'] = factors(D, market)          # built on the full history, then sliced
    F = D['_F'].loc[ls.index]
    first = S.notna().sum(axis=1).gt(0).idxmax()
    ls, lo_, F = ls.loc[first:], lo_.loc[first:], F.loc[first:]
    return ls, lo_, F, to.loc[first:], to_l.loc[first:]


def tune(market):
    import optuna
    cfg = CFG[market]
    X, D = load(market)
    cut = pd.Timestamp(cfg['train_end'], tz=TZ)
    v0, v1 = (pd.Timestamp(x, tz=TZ) for x in cfg['val'])
    step = 1 if market == 'crypto' else 5
    log = []

    def obj(trial):
        p = dict(H=trial.suggest_categorical('H', list(HS)), hold=trial.suggest_categorical('hold', [1, 3, 7]),
                 leaves=trial.suggest_int('leaves', 7, 63, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 600, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 100, 5000, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        m, cols = fit(X, p['H'], cut, p, step)
        S = predict(m, cols, X, v0, v1)
        ls, lo_, F, to, _ = evaluate_scores(S, D, market, p['hold'], v0, v1)
        a, t, _ = alpha(ls, F)
        log.append(dict(p, t=t, alpha_ann=a * cfg['bpy'], sharpe=stats(ls, cfg['bpy'])['sharpe'], turnover=float(to.mean())))
        print(trial.number, round(t, 2), round(a * cfg['bpy'], 3), round(log[-1]['sharpe'], 2), p, flush=True)
        return t

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=30)
    json.dump(dict(params=st.best_params, dev_t=st.best_value, trials=log, frozen_at=str(pd.Timestamp.utcnow())),
              open(f'FROZEN_stage16_{market}.json', 'w'), indent=1, default=str)
    print('BEST', st.best_params, st.best_value, flush=True)


def test(market):
    cfg = CFG[market]
    p = json.load(open(f'FROZEN_stage16_{market}.json'))['params']
    X, D = load(market)
    step = 1 if market == 'crypto' else 5
    parts, imp = [], []
    for Y in cfg['years']:
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
        m, cols = fit(X, p['H'], lo, p, step)
        parts.append(predict(m, cols, X, lo, hi))
        imp.append(pd.Series(m.feature_importance('gain'), index=cols, name=Y))
        print(Y, flush=True)
    S = pd.concat(parts)
    S.to_pickle(f'{CD}/{market}_scores.pkl')
    pd.concat(imp, axis=1).to_csv(f'importance_stage16_{market}.csv')
    report(market, S, D, p)


def report(market, S, D, p):
    cfg = CFG[market]
    t0, t1 = (pd.Timestamp(x, tz=TZ) for x in cfg['test'])
    S = S[(S.index.get_level_values(0) >= t0) & (S.index.get_level_values(0) < t1)]
    bpy = cfg['bpy']
    ls, lo_, F, to, tol = evaluate_scores(S, D, market, p['hold'], t0, t1)
    rows = []
    for name, x in (('long-short', ls), ('long-only', lo_)):
        a, t, m = alpha(x, F)
        rows.append(dict(market=market, book=name, alpha_ann=a * bpy, alpha_t=t, **stats(x, bpy),
                         turnover_day=float((to if name == 'long-short' else tol).mean()),
                         **{f'b_{k}': float(v) for k, v in m.params.items() if k != 'const'}))
    for k in F.columns:
        rows.append(dict(market=market, book=f'factor {k}', **stats(F[k], bpy)))
    # robustness: one extra day of delay
    S2 = S.unstack().shift(1).stack()
    ls2, _, F2, _, _ = evaluate_scores(S2, D, market, p['hold'], t0, t1)
    a2, t2, _ = alpha(ls2, F2)
    rows.append(dict(market=market, book='long-short +1 day delay', alpha_ann=a2 * bpy, alpha_t=t2, **stats(ls2, bpy)))
    R = pd.DataFrame(rows)
    yr = []
    for y, g in ls.groupby(ls.index.year):
        a, t, _ = alpha(g, F.loc[g.index])
        yr.append(dict(market=market, year=y, alpha_ann=a * bpy, t=t, **stats(g, bpy)))
    R.to_csv(f'results_stage16_{market}.csv', index=False)
    pd.DataFrame(yr).to_csv(f'results_stage16_{market}_years.csv', index=False)
    pd.concat([ls.rename('long_short'), lo_.rename('long_only'), F], axis=1).to_pickle(f'{CD}/{market}_returns.pkl')
    pd.set_option('display.width', 250)
    print(R.round(3).to_string())
    print(pd.DataFrame(yr).round(3).to_string())
    print(f'PRIMARY {market}:', bool(R.iloc[0].alpha_ann > 0 and R.iloc[0].alpha_t >= 3))


def rereport(market):
    p = json.load(open(f'FROZEN_stage16_{market}.json'))['params']
    X, D = load(market)
    report(market, pd.read_pickle(f'{CD}/{market}_scores.pkl'), D, p)


if __name__ == '__main__':
    dict(build=build, tune=tune, test=test, rereport=rereport)[sys.argv[1]](sys.argv[2])

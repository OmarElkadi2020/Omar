"""Stage 20: long-only up-trend classifier with cash vs every long-only indicator (PREREG_stage20_long_only_trend.md).
python -m tc.stage20 build|tune|test"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .stage16 import (load_crypto, daily_features, mtf_crypto, xrank, alpha, stats, run_book, params, CD, TZ, EMB,
                      rsi)
from .evaluate import supertrend
from .strat_lag import adx

HS = (3, 7, 14, 30)
COST, BPY = 0.001, 365
FOLDS = (2019, 2020, 2021)
T0, T1 = pd.Timestamp('2022-01-01', tz=TZ), pd.Timestamp('2026-09-01', tz=TZ)
TAG = 'crypto20'


# ------------------------------------------------------------------ indicators (long-only "in" masks)
def donchian_state(C, H, L, n_in=20, n_out=10):
    hi = H.rolling(n_in).max().shift(1).values
    lo = L.rolling(n_out).min().shift(1).values
    c = C.values
    st = np.zeros(c.shape, bool)
    cur = np.zeros(c.shape[1], bool)
    for i in range(len(c)):
        cur = np.where(cur, ~(c[i] < lo[i]), c[i] > hi[i])
        cur &= ~np.isnan(c[i])
        st[i] = cur
    return pd.DataFrame(st, index=C.index, columns=C.columns)


def indicators(P):
    C, H, L = P['C'], P['H'], P['L']
    I = {}
    I['EMA50'] = C > C.ewm(span=50, adjust=False).mean()
    I['EMA200'] = C > C.ewm(span=200, adjust=False).mean()
    I['GOLDEN'] = C.rolling(50).mean() > C.rolling(200).mean()
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    I['MACD'] = macd > macd.ewm(span=9, adjust=False).mean()
    I['TSMOM30'] = C > C.shift(30)
    I['TSMOM90'] = C > C.shift(90)
    I['DONCHIAN20_10'] = donchian_state(C, H, L)
    st, ad = {}, {}
    for s in C.columns:
        f = pd.DataFrame(dict(high=H[s], low=L[s], close=C[s])).dropna()
        if len(f) < 60:
            continue
        st[s] = pd.Series(supertrend(f, 10, 3.0) == 1, index=f.index)
        a = adx(f, 14)
        up, dn = f.high.diff(), -f.low.diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), f.index).ewm(alpha=1 / 14, adjust=False).mean()
        ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), f.index).ewm(alpha=1 / 14, adjust=False).mean()
        ad[s] = pd.Series((a > 25) & (pdm.values > ndm.values), index=f.index)
    I['SUPERTREND10_3'] = pd.DataFrame(st).reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
    I['ADX25_DI'] = pd.DataFrame(ad).reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
    return I


def ew_in(mask):
    w = mask.astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def benchmark_books(D):
    elig, C = D['elig'], D['C'].astype(float)
    B = {k: ew_in(v & elig) for k, v in D['ind'].items()}
    r30 = np.log(C / C.shift(30)).where(elig)
    rk = r30.rank(axis=1, pct=True)
    top = rk > 0.8
    B['XS_MOM_TOP20'] = ew_in(top)
    B['DUAL_MOM'] = (top & (r30 > 0)).astype(float).div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    B['EW_UNIVERSE'] = ew_in(elig)
    btc = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    btc['BTCUSDT'] = 1.0
    B['BTC_BH'] = btc
    b2 = btc.copy()
    b2['BTCUSDT'] = D['ind']['EMA50']['BTCUSDT'].astype(float)
    B['BTC_EMA50'] = b2
    return B


# ------------------------------------------------------------------ build
def build():
    P, P4 = load_crypto()
    C = P['C']
    r = np.log(C).diff().clip(-1, 1)
    hist = C.notna().cumsum()
    elig = (hist >= 60) & C.notna() & P['EX'].notna()
    dvm = P['DV'].rolling(30, min_periods=10).median().where(elig)
    elig = elig & (dvm.rank(axis=1, ascending=False) <= 100)
    mkt = r.where(elig).mean(axis=1)
    F = daily_features(P, mkt, (1, 3, 7, 14, 30, 60, 90, 180))
    F.update(mtf_crypto(P4, C.index))
    cols = []
    for k in list(F):
        v = F.pop(k)
        cols.append(xrank(v, elig).where(elig).stack().dropna().rename('rk_' + k).astype(np.float32))
        if k not in ('size', 'age'):
            cols.append(v.where(elig).stack().dropna().rename('raw_' + k).astype(np.float32))
    lex = np.log(P['EX'])
    for H in HS:
        cols.append((lex.shift(-H) - lex).where(elig).stack().dropna().rename(f'fwd{H}').astype(np.float32))
    X = pd.concat(cols, axis=1)
    X.index.names = ['date', 'asset']
    X = X[X.index.get_level_values(1).isin(C.columns)]
    lm = mkt.fillna(0).cumsum()
    lb = np.log(C['BTCUSDT'])
    rb = lb.diff()
    ctx = pd.DataFrame({f'mkt_ret_{n}': lm - lm.shift(n) for n in (7, 30, 90)})
    for n in (7, 30, 90):
        ctx[f'btc_ret_{n}'] = lb - lb.shift(n)
    ctx['btc_ema_d_100'] = (lb - np.log(C['BTCUSDT'].ewm(span=100, adjust=False).mean())) / rb.rolling(30).std()
    ctx['btc_vol_30'] = rb.rolling(30).std()
    ctx['btc_rsi_14'] = rsi(C['BTCUSDT'])
    r30 = np.log(C) - np.log(C.shift(30))
    ctx['breadth_30'] = (r30.where(elig) > 0).sum(axis=1) / elig.sum(axis=1)
    ctx['breadth_ema50'] = ((C > C.ewm(span=50, adjust=False).mean()) & elig).sum(axis=1) / elig.sum(axis=1)
    ctx['disp_30'] = r30.where(elig).std(axis=1)
    X = X.join(ctx.astype(np.float32), on='date')
    X.to_pickle(f'{CD}/{TAG}_X.pkl')
    ind = indicators(P)
    rex = (lex.shift(-1) - lex).clip(-1, 1)
    pd.to_pickle(dict(rex=rex.astype(np.float32), elig=elig, C=C.astype(np.float32), ind=ind), f'{CD}/{TAG}_P.pkl')
    print(X.shape, flush=True)


def load():
    X = pd.read_pickle(f'{CD}/{TAG}_X.pkl')
    D = pd.read_pickle(f'{CD}/{TAG}_P.pkl')
    D['rex'] = np.expm1(D['rex'].astype(float))
    return X, D


_B = {}


def bench_returns(D):
    if 'R' not in _B:
        B = benchmark_books(D)
        _B['B'] = B
        _B['R'] = pd.DataFrame({k: run_book(w, D['rex'], COST, 1)[0] for k, w in B.items()})
    return _B['R'], _B['B']


# ------------------------------------------------------------------ model
def feat_cols(X):
    return [c for c in X.columns if not c.startswith('fwd')]


def fit(X, H, cut, p):
    d = X.index.get_level_values(0)
    ok = (d + pd.Timedelta(days=int(H * 1.5) + 2) < cut - EMB) & X[f'fwd{H}'].notna().values
    t = X[ok]
    cols = feat_cols(X)
    y = (t[f'fwd{H}'].values > 0).astype(float)
    w = np.abs(t[f'fwd{H}'].values) + 1e-4
    pr = params(p)
    pr['objective'] = 'binary'
    m = lgb.train(pr, lgb.Dataset(t[cols].values, y, weight=w, feature_name=cols), num_boost_round=p['trees'])
    return m, cols


def predict(m, cols, X, lo, hi):
    d = X.index.get_level_values(0)
    w = (d >= lo) & (d < hi)
    return pd.Series(m.predict(X.loc[w, cols].values), index=X.index[w])


def model_book(P_, K, tau):
    rk = P_.where(P_ > tau).rank(axis=1, ascending=False, method='first')
    return (rk <= K).astype(float) / K


def model_returns(Pr, D, p, lo, hi):
    rex = D['rex'].loc[lo:hi]
    P_ = Pr.unstack().reindex(index=rex.index)
    w = model_book(P_, p['K'], p['tau'])
    r, to = run_book(w, rex, COST, p['hold'])
    return r, to, w


def evaluate(r, R):
    Rb = R.loc[r.index]
    a, t, _ = alpha(r, Rb)
    return a, t


def tune():
    import optuna
    X, D = load()
    R, _ = bench_returns(D)
    log = []

    def obj(trial):
        p = dict(H=trial.suggest_categorical('H', list(HS)), K=trial.suggest_categorical('K', [3, 5, 10, 20]),
                 tau=trial.suggest_categorical('tau', [0.5, 0.55, 0.6, 0.65]),
                 hold=trial.suggest_categorical('hold', [1, 3, 7]),
                 leaves=trial.suggest_int('leaves', 7, 63, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 600, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 100, 3000, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        ts, shs = [], []
        for Y in FOLDS:
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
            m, cols = fit(X, p['H'], lo, p)
            r, _, _ = model_returns(predict(m, cols, X, lo, hi), D, p, lo, hi)
            a, t = evaluate(r, R)
            ts.append(t)
            shs.append(stats(r, BPY)['sharpe'])
        v = float(np.mean(ts))
        log.append(dict(p, t_mean=v, t_folds=ts, sharpe_folds=shs))
        print(trial.number, round(v, 2), [round(x, 2) for x in ts], [round(x, 2) for x in shs], p, flush=True)
        return v

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=40)
    json.dump(dict(params=st.best_params, dev_t=st.best_value, trials=log, frozen_at=str(pd.Timestamp.utcnow())),
              open('FROZEN_stage20.json', 'w'), indent=1, default=str)
    print('BEST', st.best_params, st.best_value, flush=True)


def classification_quality(w, D, X, H, lo, hi):
    fwd = X[f'fwd{H}'].unstack().reindex(index=w.index, columns=w.columns)
    held = (w > 0) & fwd.notna()
    f = fwd.where(held).stack().dropna()
    f = np.expm1(f)
    return dict(buy_precision=float((f > 0).mean()), mean_fwd=float(f.mean()), false_buy_rate=float((f < -0.05).mean()),
                time_in_market=float((w.sum(axis=1) > 0.001).mean()), avg_exposure=float(w.sum(axis=1).mean()),
                n_held_days=int(len(f)))


def test():
    p = json.load(open('FROZEN_stage20.json'))['params']
    X, D = load()
    R, B = bench_returns(D)
    parts = []
    for Y in range(T0.year, T1.year + 1):
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), min(pd.Timestamp(f'{Y + 1}-01-01', tz=TZ), T1)
        m, cols = fit(X, p['H'], lo, p)
        parts.append(predict(m, cols, X, lo, hi))
        print(Y, flush=True)
    Pr = pd.concat(parts)
    Pr.to_pickle(f'{CD}/{TAG}_probs.pkl')
    r, to, w = model_returns(Pr, D, p, T0, T1)
    Rb = R.loc[r.index]
    a, t, m = alpha(r, Rb)
    rows = [dict(strategy='MODEL (stage 20)', alpha_ann=a * BPY, alpha_t=t, turnover_day=float(to.mean()), **stats(r, BPY),
                 **classification_quality(w.rolling(p['hold'], min_periods=1).mean(), D, X, p['H'], T0, T1))]
    for k in Rb.columns:
        rows.append(dict(strategy=k, **stats(Rb[k], BPY),
                         **classification_quality(B[k].loc[T0:T1], D, X, p['H'], T0, T1)))
    T = pd.DataFrame(rows)
    T.to_csv('results_stage20_crypto.csv', index=False)
    C = pd.concat([r.rename('MODEL'), Rb], axis=1)
    yr = C.groupby(C.index.year).apply(lambda g: np.expm1(np.log1p(g.clip(lower=-0.99)).sum()))
    yr.to_csv('results_stage20_crypto_years.csv')
    C.to_pickle(f'{CD}/{TAG}_returns.pkl')
    pd.set_option('display.width', 260)
    print(T.round(3).to_string())
    print(yr.round(2).to_string())
    print('betas:', {k: round(v, 2) for k, v in m.params.items()})
    ok1 = a > 0 and t >= 3.0
    ok2 = T.sharpe.iloc[0] > T.sharpe.iloc[1:].max()
    print('P1 alpha t>=3:', ok1, ' P2 Sharpe > every benchmark:', ok2, ' PRIMARY:', bool(ok1 and ok2))


if __name__ == '__main__':
    dict(build=build, tune=tune, test=test)[sys.argv[1]]()

"""Stage 17: stocks, monthly residual-trend model (see PREREG_stage17_stocks_monthly_residual_trend.md).
python -m tc.stage17 build|tune|test"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .stage16 import (load_stocks, load_india, xrank, book, run_book, alpha, stats, params, CD, TZ, EMB, factors, CFG)

H, HOLD, BPY = 21, 21, 252
MARKET = 'stocks'
COST = 0.0005
LONG_ONLY_PRIMARY = False
TAG = 'stocks17'
FROZEN = 'FROZEN_stage17.json'
OUT = 'stage17_stocks'
MAS = (3, 5, 10, 20, 50, 100, 200, 400)
TRAIN_END = pd.Timestamp('2006-01-01', tz=TZ)
V0, V1 = pd.Timestamp('2006-01-01', tz=TZ), pd.Timestamp('2011-01-01', tz=TZ)
T0, T1 = pd.Timestamp('2011-01-01', tz=TZ), pd.Timestamp('2026-09-23', tz=TZ)


def roll_beta(r, m, n):
    cov = r.rolling(n, min_periods=n // 2).cov(m)
    return cov.div(m.rolling(n, min_periods=n // 2).var(), axis=0)


def build():
    X = pd.read_pickle(f'{CD}/{MARKET}_X.pkl')
    D = pd.read_pickle(f'{CD}/{MARKET}_P.pkl')
    P = load_stocks() if MARKET == 'stocks' else load_india()
    elig = D['elig']
    C = P['C'].reindex(index=elig.index, columns=elig.columns)
    Hh = P['H'].reindex(index=elig.index, columns=elig.columns)
    r = np.log(C).diff().clip(-1, 1)
    mkt = r.where(elig).mean(axis=1)
    F = {}
    for L in MAS:
        F[f'ma_{L}'] = np.log(C / C.rolling(L, min_periods=max(2, L // 2)).mean())
    F['hi52'] = np.log(C / Hh.rolling(252, min_periods=126).max())
    b252 = roll_beta(r, mkt, 252)
    res = r - b252.mul(mkt, axis=0)
    rs = res.rolling(252, min_periods=126).std()
    F['resmom_12_1'] = res.shift(21).rolling(231, min_periods=120).sum() / rs
    F['resmom_6_1'] = res.shift(21).rolling(105, min_periods=60).sum() / rs
    F['ivol_60'] = res.rolling(60, min_periods=30).std()
    new = []
    for k in list(F):
        new.append(xrank(F.pop(k), elig).stack().dropna().rename(k).astype(np.float32))
    # residual forward target from the execution price
    E = np.log(P['EX'].reindex(index=elig.index, columns=elig.columns))
    fwd = E.shift(-H) - E
    mfwd = fwd.where(elig).mean(axis=1)
    y = xrank(fwd - b252.mul(mfwd, axis=0), elig & fwd.notna())
    new.append(y.stack().dropna().rename('y21r').astype(np.float32))
    lm = mkt.fillna(0).cumsum()
    ctx = pd.DataFrame({'mkt_ret_504': lm - lm.shift(504), 'mkt_vol_126': mkt.rolling(126).std()}).astype(np.float32)
    N = pd.concat(new, axis=1)
    N.index.names = ['date', 'asset']
    X = X.drop(columns=[c for c in X.columns if c.startswith('y')]).join(N, how='left').join(ctx, on='date')
    X.to_pickle(f'{CD}/{TAG}_X.pkl')
    trend = pd.concat([xrank(np.log(C / C.rolling(L, min_periods=max(2, L // 2)).mean()), elig) for L in MAS]).groupby(level=0).mean()
    trend.astype(np.float32).to_pickle(f'{CD}/{TAG}_trend.pkl')
    print(X.shape, X['y21r'].notna().sum(), flush=True)


def load():
    X = pd.read_pickle(f'{CD}/{TAG}_X.pkl')
    D = pd.read_pickle(f'{CD}/{MARKET}_P.pkl')
    D['rex'] = np.expm1(D['rex'].astype(float))
    return X, D


def vol_manage(x, target=0.10, n=126, cap=2.0):
    s = x.rolling(n, min_periods=60).std().shift(1) * np.sqrt(BPY)
    return (x * np.minimum(cap, target / s)).dropna()


_F = {}


def controls(D):
    if 'F' not in _F:
        F = factors(D, MARKET)
        F['UMD_VM'] = vol_manage(F['UMD'])
        trend = pd.read_pickle(f'{CD}/{TAG}_trend.pkl').reindex(D['rex'].index)
        F['TREND'], _ = run_book(book(trend.where(D['elig']), q=1 / 3), D['rex'], 0.0)
        _F['F'] = F
    return _F['F']


def feat_cols(X):
    return [c for c in X.columns if c != 'y21r']


def fit(X, cut, p):
    d = X.index.get_level_values(0)
    ok = (d + pd.Timedelta(days=int(H * 1.5) + 2) < cut - EMB) & X['y21r'].notna().values
    t = X[ok]
    dd = t.index.get_level_values(0)
    t = t[((dd - dd.min()).days % 5) == 0]
    cols = feat_cols(X)
    m = lgb.train(params(p), lgb.Dataset(t[cols].values, t['y21r'].values, feature_name=cols), num_boost_round=p['trees'])
    return m, cols


def predict(m, cols, X, lo, hi):
    d = X.index.get_level_values(0)
    w = (d >= lo) & (d < hi)
    return pd.Series(m.predict(X.loc[w, cols].values), index=X.index[w])


def books(S, D, lo, hi, delay=0):
    rex = D['rex'].loc[lo:hi]
    S = S.unstack().reindex(index=rex.index)
    if delay:
        S = S.shift(delay)
    ls, to = run_book(book(S), rex, COST, HOLD)
    first = S.notna().sum(axis=1).gt(0).idxmax()
    return S, rex, ls.loc[first:], to.loc[first:]


def tune():
    import optuna
    X, D = load()
    F = controls(D)
    log = []

    def obj(trial):
        p = dict(leaves=trial.suggest_int('leaves', 7, 63, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 600, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 200, 5000, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        m, cols = fit(X, TRAIN_END, p)
        S = predict(m, cols, X, V0, V1)
        Sw, rex, ls, _ = books(S, D, V0, V1)
        if LONG_ONLY_PRIMARY:
            vm = run_book(book(Sw, long_only=True), rex, COST, HOLD)[0].loc[ls.index[0]:]
        else:
            vm = vol_manage(ls)      # first ~60 days dropped: the vol estimate must come from the strictly past book
        a, t, _ = alpha(vm, F.loc[vm.index])
        log.append(dict(p, t=t, alpha_ann=a * BPY, sharpe=stats(vm, BPY)['sharpe']))
        print(trial.number, round(t, 2), round(a * BPY, 3), round(log[-1]['sharpe'], 2), p, flush=True)
        return t

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=20)
    json.dump(dict(params=st.best_params, dev_t=st.best_value, trials=log, frozen_at=str(pd.Timestamp.utcnow())),
              open(FROZEN, 'w'), indent=1, default=str)
    print('BEST', st.best_params, st.best_value, flush=True)


def test():
    from .stage10 import splits
    p = json.load(open(FROZEN))['params']
    X, D = load()
    F = controls(D)
    parts = []
    for Y in range(2011, 2027):
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
        m, cols = fit(X, lo, p)
        parts.append(predict(m, cols, X, lo, hi))
        print(Y, flush=True)
    S = pd.concat(parts)
    S.to_pickle(f'{CD}/{TAG}_scores.pkl')
    rows = []

    def add(name, x, extra=None):
        a, t, m = alpha(x, F.loc[x.index])
        rows.append(dict(book=name, alpha_ann=a * BPY, alpha_t=t, **stats(x, BPY), **(extra or {}),
                         **{f'b_{k}': float(v) for k, v in m.params.items() if k != 'const'}))

    Sw, rex, ls, to = books(S, D, T0 - pd.Timedelta('400D'), T1)   # scores before T0 are NaN; warm-up only
    ls = ls.loc[T0:]
    vm = vol_manage(ls)
    lo_only = run_book(book(Sw, long_only=True), rex, COST, HOLD)[0].loc[T0:]
    if LONG_ONLY_PRIMARY:
        add('PRIMARY long-only top 20%', lo_only)
        add('vol-managed long-short', vm, dict(turnover_day=float(to.loc[T0:].mean())))
        add(f'long-only, costs {int(COST * 1e4) + 10}bp', run_book(book(Sw, long_only=True), rex, COST + 0.001, HOLD)[0].loc[T0:])
        add('long-only +1 day delay', run_book(book(Sw.shift(1), long_only=True), rex, COST, HOLD)[0].loc[T0:])
        vm = lo_only
    else:
        add('PRIMARY vol-managed long-short', vm, dict(turnover_day=float(to.loc[T0:].mean())))
    add('unscaled long-short', ls)
    add('long-only top 20%', run_book(book(Sw, long_only=True), rex, COST, HOLD)[0].loc[T0:])
    add('costs 10bp (unscaled)', run_book(book(Sw), rex, 0.001, HOLD)[0].loc[T0:])
    dv = D['dv'].loc[rex.index]
    for k in (200, 50):
        Sk = Sw.where(dv.where(Sw.notna()).rank(axis=1, ascending=False) <= k)
        add(f'top-{k} by dollar volume (unscaled)', run_book(book(Sk), rex, COST, HOLD)[0].loc[T0:])
    add('+1 day delay (unscaled)', run_book(book(Sw.shift(1)), rex, COST, HOLD)[0].loc[T0:])
    R = pd.DataFrame(rows)
    yr = []
    for y, g in vm.groupby(vm.index.year):
        a, t, _ = alpha(g, F.loc[g.index])
        yr.append(dict(year=y, alpha_ann=a * BPY, t=t, **stats(g, BPY)))
    R.to_csv(f'results_{OUT}.csv', index=False)
    pd.DataFrame(yr).to_csv(f'results_{OUT}_years.csv', index=False)
    pd.concat([vm.rename('vm_long_short'), ls.rename('long_short'), F.loc[T0:]], axis=1).to_pickle(f'{CD}/{TAG}_returns.pkl')
    pd.set_option('display.width', 250)
    print(R.round(3).to_string())
    print(pd.DataFrame(yr).round(3).to_string())
    print({k: round(stats(F[k].loc[T0:], BPY)['sharpe'], 2) for k in F})
    mk = F['MKT'].loc[T0:]
    print('EW market', stats(mk, BPY), ' long-only', stats(lo_only, BPY))
    print('PRIMARY:', bool(R.iloc[0].alpha_ann > 0 and R.iloc[0].alpha_t >= (3.0 if LONG_ONLY_PRIMARY else 3.5)))


if __name__ == '__main__':
    dict(build=build, tune=tune, test=test)[sys.argv[1]]()

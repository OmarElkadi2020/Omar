"""Stage 15: forward-return trend model, alpha vs BH/TSMOM/EMA50/BAZ (see PREREG_stage15_trend_alpha.md).
python -m tc.stage15 build|tune|walk|eval"""
import json
import os
import sys
import zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from multiprocessing import Pool
from .stage10 import splits
from .stage14 import frame as frame14, market as market14, ohlcv, COINS
from .dataset import load_1h

CD = 'cache_stage15'
S12 = 'cache_stage12'
COLS = json.load(open('FROZEN_stage12.json'))['cols']
HS = (10, 20, 40)
BENCH = ['bh', 'tsmom', 'ema50', 'baz']
COST = {'stk': 0.0005, 'c1D': 0.001, 'c4h': 0.001}
LAG = {'stk': 20, 'c1D': 20, 'c4h': 120}
DEV_TRAIN_END, DEV0, DEV1 = pd.Timestamp('2006-01-01', tz='UTC'), pd.Timestamp('2006-01-01', tz='UTC'), \
    pd.Timestamp('2011-01-01', tz='UTC')
EMB = pd.Timedelta('7D')
YEARS = list(range(2011, 2027))
CRYPTO_T0 = pd.Timestamp('2021-01-01', tz='UTC')
END = pd.Timestamp('2026-09-24', tz='UTC')


def phi(x):
    return x * np.exp(-x ** 2 / 4) / 0.89


def targets(f, px):
    """forward vol-normalised returns, next-bar return, benchmark positions. Only y*/r1/fend* look ahead."""
    lr = np.log(px).diff().clip(-0.4, 0.4)
    sig = lr.ewm(span=60, min_periods=60).std()
    cum = lr.fillna(0).cumsum()
    out = pd.DataFrame(index=f.index)
    out['r1'] = lr.shift(-1)
    idx = pd.Series(f.index, index=f.index)
    for H in HS:
        out[f'y{H}'] = ((cum.shift(-H) - cum) / (sig * np.sqrt(H))).clip(-4, 4)
        out[f'fend{H}'] = idx.shift(-H)
    c = f.close
    out['tsmom'] = np.sign(np.log(c / c.shift(252))).fillna(0)
    out['ema50'] = np.sign(c - c.ewm(span=50, adjust=False).mean())
    out['baz'] = sum(phi(f[k].astype(float)) for k in ('macdn_8_24', 'macdn_16_48', 'macdn_32_96')) / 3
    out['bh'] = 1.0
    return out


def _stock(tk):
    f = pd.read_pickle(f'{S12}/{tk}.pkl')
    px = ADJ.get(tk)
    if px is None:
        return None
    px = px.reindex(f.index)
    px = px.where(px > 0)
    T = targets(f, px)
    T.to_pickle(f'{CD}/stk/{tk}.pkl')
    off = zlib.crc32(tk.encode()) % 5
    sub = pd.concat([f[COLS].iloc[off::5], T.iloc[off::5]], axis=1)
    sub['asset'] = tk
    return sub


def build():
    global ADJ
    os.makedirs(f'{CD}/stk', exist_ok=True)
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'adj_close'])
    d['date'] = pd.to_datetime(d.date).dt.tz_localize('UTC')
    ADJ = {t: g.set_index('date').adj_close.sort_index().pipe(lambda s: s[~s.index.duplicated()])
           for t, g in d.groupby('ticker')}
    tks = sorted(x[:-4] for x in os.listdir(S12))
    with Pool(4) as p:
        subs = [s for s in p.imap(_stock, tks, chunksize=8) if s is not None]
    S = pd.concat(subs)
    S.to_pickle(f'{CD}/stocks_sub.pkl')
    print('stocks', len(subs), S.shape, flush=True)
    for tf in ('4h', '1D'):
        mk = market14({a: ohlcv(load_1h, a, tf).close for a in COINS})
        for a in COINS:
            f = frame14(load_1h, a, tf, mk)
            f = pd.concat([f[['open', 'high', 'low', 'close']], f[COLS]], axis=1)
            T = targets(f, f.close)
            pd.concat([f, T], axis=1).to_pickle(f'{CD}/c{tf}_{a}.pkl')
        print('crypto', tf, flush=True)


# ---------------------------------------------------------------- portfolio + alpha
def positions(pred, s, span):
    p = pd.Series(pred)
    if span > 1:
        p = p.ewm(span=span, adjust=False).mean()
    return np.clip(p.values / s, -1, 1)


def pnl(pos, r1, cost):
    pos = np.nan_to_num(np.asarray(pos, float))
    return pos * r1 - cost * np.abs(np.diff(pos, prepend=0.0))


def asset_returns(T, pos, cost, lo, hi):
    """per-bar net returns of model + benchmarks for one asset, restricted to [lo, hi) with a valid r1."""
    w = (T.index >= lo) & (T.index < hi) & T.r1.notna().values & ~np.isnan(pos)
    r1 = T.r1.fillna(0).values
    out = {'model': pnl(np.where(np.isnan(pos), 0, pos), r1, cost)}
    for b in BENCH:
        out[b] = pnl(T[b].fillna(0).values, r1, cost)
    out['model_lf'] = pnl(np.clip(np.where(np.isnan(pos), 0, pos), 0, 1), r1, cost)
    return pd.DataFrame(out, index=T.index)[w]


def span_alpha(P, lag, cols=BENCH):
    X = sm.add_constant(P[cols])
    m = sm.OLS(P['model'], X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return float(m.params['const']), float(m.tvalues['const']), m


def ew(parts):
    """equal-weight portfolio from a list of per-asset return frames."""
    cat = pd.concat(parts, keys=range(len(parts)))
    return cat.groupby(level=1).mean().sort_index()


def sharpe(x, bpy):
    return float(x.mean() / x.std() * np.sqrt(bpy)) if x.std() > 0 else 0.0


# ---------------------------------------------------------------- model
def lgbp(p):
    return dict(objective='regression', learning_rate=p['lr'], num_leaves=p['leaves'],
                min_data_in_leaf=p['min_leaf'], feature_fraction=p['ff'], bagging_fraction=p['bf'],
                bagging_freq=1, lambda_l2=p['l2'], verbose=-1, num_threads=4, seed=0, deterministic=True)


def fit(X, y, w, p):
    ds = lgb.Dataset(X.values.astype(np.float32), y, weight=w, feature_name=COLS, free_raw_data=True)
    m = lgb.train(lgbp(p), ds, num_boost_round=p['trees'])
    s = float(np.std(m.predict(X.values.astype(np.float32))))
    return m, s


def tune():
    import optuna
    S = pd.read_pickle(f'{CD}/stocks_sub.pkl')
    sp = splits()
    tr = S[S.asset.isin(sp['fit'])]
    val = []
    for tk in sp['val']:
        if not os.path.exists(f'{CD}/stk/{tk}.pkl'):
            continue
        f = pd.read_pickle(f'{S12}/{tk}.pkl')
        T = pd.read_pickle(f'{CD}/stk/{tk}.pkl')
        w = (f.index >= DEV0 - pd.Timedelta('400D')) & (f.index < DEV1)
        if w.sum() > 300:
            val.append((f.loc[w, COLS].values.astype(np.float32), T[w]))
    print('val tickers', len(val), flush=True)
    log = []

    def objective(trial):
        p = dict(H=trial.suggest_categorical('H', list(HS)),
                 span=trial.suggest_categorical('span', [1, 3, 5, 10]),
                 leaves=trial.suggest_int('leaves', 7, 63, log=True),
                 lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 600, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 200, 5000, log=True),
                 ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0),
                 l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        H = p['H']
        t = tr[(tr[f'fend{H}'] < DEV_TRAIN_END - EMB) & tr[f'y{H}'].notna()]
        m, s = fit(t[COLS], t[f'y{H}'].values, None, p)
        parts = []
        for X, T in val:
            pos = positions(m.predict(X), s, p['span'])
            parts.append(asset_returns(T, pos, COST['stk'], DEV0, DEV1))
        P = ew(parts)
        a, tv, _ = span_alpha(P, LAG['stk'])
        log.append(dict(p, alpha_t=tv, alpha_ann=a * 252, sh=sharpe(P.model, 252)))
        print(trial.number, round(tv, 2), round(a * 252, 4), p, flush=True)
        return tv

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, n_trials=40)
    best = dict(st.best_params)
    json.dump(dict(params=best, dev_alpha_t=st.best_value, trials=log, cols=COLS,
                   frozen_at=str(pd.Timestamp.utcnow())), open('FROZEN_stage15.json', 'w'), indent=1, default=str)
    print('BEST', best, st.best_value)


# ---------------------------------------------------------------- walk-forward
def crypto_frames():
    out = {}
    for tf in ('4h', '1D'):
        for a in COINS:
            out[(f'c{tf}', a)] = pd.read_pickle(f'{CD}/c{tf}_{a}.pkl')
    return out


def walk():
    p = json.load(open('FROZEN_stage15.json'))['params']
    H = p['H']
    os.makedirs(f'{CD}/models', exist_ok=True)
    S = pd.read_pickle(f'{CD}/stocks_sub.pkl')
    S = S[S[f'y{H}'].notna()]
    C = crypto_frames()
    csub = []
    for (g, a), f in C.items():
        x = f.iloc[zlib.crc32(a.encode()) % 5::5] if g == 'c4h' else f
        x = x[x[f'y{H}'].notna()].copy()
        x['grp'] = g
        csub.append(x[COLS + [f'y{H}', f'fend{H}', 'grp']])
    CS = pd.concat(csub)
    for Y in YEARS:
        fn = f'{CD}/models/m{Y}.txt'
        if os.path.exists(fn):
            continue
        cut = pd.Timestamp(f'{Y}-01-01', tz='UTC') - EMB
        s = S[S[f'fend{H}'] < cut]
        c = CS[CS[f'fend{H}'] < cut]
        X = pd.concat([s[COLS], c[COLS]])
        y = np.r_[s[f'y{H}'].values, c[f'y{H}'].values]
        w = np.ones(len(X))
        n = len(X)
        for g in ('c4h', 'c1D'):
            k = (c.grp == g).values
            if k.sum():
                w[len(s):][k] = 0.125 * n / k.sum()
        k = np.zeros(n, bool)
        k[:len(s)] = True
        w[k] = (n - w[~k].sum()) / k.sum()
        m, sd = fit(X, y, w, p)
        m.save_model(fn)
        json.dump(dict(s=sd, n=int(n), n_crypto=int(len(c))), open(f'{CD}/models/m{Y}.json', 'w'))
        print(Y, n, len(c), round(sd, 4), flush=True)


def _pred_stock(tk):
    p = json.load(open('FROZEN_stage15.json'))['params']
    f = pd.read_pickle(f'{S12}/{tk}.pkl')
    X = f[COLS].values.astype(np.float32)
    pred = np.full(len(f), np.nan)
    for Y in YEARS:
        w = (f.index >= pd.Timestamp(f'{Y}-01-01', tz='UTC')) & (f.index < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))
        if w.any():
            pred[w] = MODELS[Y].predict(X[w]) / SD[Y]
    return tk, pd.Series(pred, index=f.index, dtype=np.float32)


def _load_models():
    global MODELS, SD
    MODELS = {Y: lgb.Booster(model_file=f'{CD}/models/m{Y}.txt') for Y in YEARS}
    SD = {Y: json.load(open(f'{CD}/models/m{Y}.json'))['s'] for Y in YEARS}


def predict_all():
    _load_models()
    tks = sorted(x[:-4] for x in os.listdir(f'{CD}/stk'))
    stk = dict(_pred_stock(t) for t in tks)     # sequential: forking after loading boosters deadlocks OpenMP
    C = crypto_frames()
    cr = {}
    for key, f in C.items():
        X = f[COLS].values.astype(np.float32)
        pred = np.full(len(f), np.nan)
        for Y in YEARS:
            w = (f.index >= pd.Timestamp(f'{Y}-01-01', tz='UTC')) & (f.index < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))
            if w.any():
                pred[w] = MODELS[Y].predict(X[w]) / SD[Y]
        cr[key] = pd.Series(pred, index=f.index)
    pd.to_pickle(dict(stk=stk, cr=cr), f'{CD}/preds.pkl')


def per_asset_alpha(parts, lag):
    rows = []
    for name, R in parts.items():
        if len(R) < 250 or R[BENCH[1:]].std().min() == 0:
            continue
        a, t, _ = span_alpha(R, lag)
        a1, t1, _ = span_alpha(R, lag, ['bh'])
        rows.append(dict(asset=name, alpha=a, t=t, alpha_bh=a1, t_bh=t1))
    return pd.DataFrame(rows)


def report(group, parts, lag, bpy, lo):
    from scipy.stats import binomtest
    P = ew(list(parts.values()))
    a, t, m = span_alpha(P, lag)
    a1, t1, _ = span_alpha(P, lag, ['bh'])
    row = dict(group=group, n_assets=len(parts), start=str(P.index[0].date()), end=str(P.index[-1].date()),
               alpha_ann=a * bpy, alpha_t=t, alpha_bh_ann=a1 * bpy, alpha_bh_t=t1,
               **{f'beta_{k}': float(m.params[k]) for k in BENCH},
               **{f'sharpe_{k}': sharpe(P[k], bpy) for k in ['model', 'model_lf'] + BENCH})
    PA = per_asset_alpha(parts, lag)
    k = int((PA.alpha > 0).sum())
    row.update(assets_alpha_pos=f'{k}/{len(PA)}', assets_alpha_p=binomtest(k, len(PA)).pvalue,
               assets_t_gt2=int((PA.t > 2).sum()), assets_t_lt_m2=int((PA.t < -2).sum()))
    yr = []
    for y, g in P.groupby(P.index.year):
        if len(g) > 100:
            ay, ty, _ = span_alpha(g, lag)
            yr.append(dict(group=group, year=y, alpha_ann=ay * bpy, t=ty))
    return row, PA.assign(group=group), yr


def evaluate():
    p = json.load(open('FROZEN_stage15.json'))['params']
    H, span = p['H'], p['span']
    PR = pd.read_pickle(f'{CD}/preds.pkl')
    T0 = pd.Timestamp('2011-01-01', tz='UTC')
    parts, ic = {}, []
    for tk, pred in PR['stk'].items():
        T = pd.read_pickle(f'{CD}/stk/{tk}.pkl')
        pos = positions(pred.values, 1.0, span)
        R = asset_returns(T, pos, COST['stk'], T0, END)
        if len(R) > 20:
            parts[tk] = R
        ic.append(pd.DataFrame({'pred': pred, 'y': T[f'y{H}'], 'tk': tk})[lambda d: d.index >= T0])
    rows, pas, yrs = [], [], []
    for name, sub in (('Stocks all (1D)', parts), ('Stocks top-50 A (1D)', {k: v for k, v in parts.items()
                                                                             if k in set(splits()['A'])})):
        r, pa, y = report(name, sub, LAG['stk'], 252, T0)
        rows.append(r); pas.append(pa); yrs += y
    C = crypto_frames()
    for g, bpy in (('c4h', 6 * 365), ('c1D', 365)):
        cp = {}
        for a in COINS:
            f = C[(g, a)]
            pos = positions(PR['cr'][(g, a)].values, 1.0, span)
            R = asset_returns(f, pos, COST[g], CRYPTO_T0, END)
            if len(R) > 20:
                cp[a] = R
        r, pa, y = report(f'Crypto {g[1:]}', cp, LAG[g], bpy, CRYPTO_T0)
        rows.append(r); pas.append(pa); yrs += y
    # cross-sectional rank IC (stocks), NW t over the daily IC series
    D = pd.concat(ic).dropna()
    D = D[D.groupby(level=0).tk.transform('size') >= 50]
    icd = D.groupby(level=0).apply(lambda g: g.pred.rank().corr(g.y.rank()))
    mi = sm.OLS(icd.values, np.ones(len(icd))).fit(cov_type='HAC', cov_kwds={'maxlags': H})
    R = pd.DataFrame(rows)
    R['xs_rank_ic'] = [float(icd.mean()), None, None, None]
    R['xs_rank_ic_t'] = [float(mi.tvalues[0]), None, None, None]
    R.to_csv('results_stage15_alpha.csv', index=False)
    pd.concat(pas).to_csv('results_stage15_per_asset.csv', index=False)
    pd.DataFrame(yrs).to_csv('results_stage15_per_year.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.T.to_string())
    print(pd.DataFrame(yrs).pivot(index='year', columns='group', values='t').round(2).to_string())
    p1 = R.iloc[0].alpha_ann > 0 and R.iloc[0].alpha_t >= 3
    p2 = all(R.iloc[i].alpha_ann > 0 and R.iloc[i].alpha_t >= 2 for i in (2, 3))
    print('P1 stocks:', p1, ' P2 crypto:', p2)


if __name__ == '__main__':
    dict(build=build, tune=tune, walk=walk, predict=predict_all, eval=evaluate)[sys.argv[1]]()

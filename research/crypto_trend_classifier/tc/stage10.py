"""Stage 10: one trend model for any single stock (see PREREG_stage10_stock_model.md).
python -m tc.stage10 build | tune | final"""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from multiprocessing import Pool
from sklearn.metrics import matthews_corrcoef
from .features import build, build_bb
from .labels import oracle_labels
from .stage3 import STOCKS
from .evaluate import smooth_state, metrics, BASELINE_FN, BASELINE_GRID, price_ema

CD = 'cache_stage10'
NON_FEAT = {'open', 'high', 'low', 'close', 'y', 'final', 'asset', 'tf'}
TUNE_CUT, DEV_END = pd.Timestamp('2006-01-01', tz='UTC'), pd.Timestamp('2011-01-01', tz='UTC')


def load_all():
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    d['date'] = pd.to_datetime(d.date).dt.tz_localize('UTC')
    return d


def market(d):
    P = d.pivot_table(index='date', columns='ticker', values='close').sort_index()
    P = P.where(P > 0)
    R = np.log(P).diff().clip(-0.4, 0.4)
    idx = np.exp(R.mean(axis=1).fillna(0).cumsum())
    m = pd.DataFrame(index=P.index)
    for n in (21, 63, 252):
        m[f'mkt_mom_{n}'] = np.log(idx / idx.shift(n))
    m['mkt_ema_dist_200'] = np.log(idx / idx.ewm(span=200, adjust=False).mean())
    s200 = P.rolling(200, min_periods=200).mean()
    m['mkt_breadth_200'] = (P > s200).sum(axis=1) / s200.notna().sum(axis=1).replace(0, np.nan)
    return m


def chop_features(b):
    """choppiness / range-bound detectors (all causal rolling windows ending at t)."""
    h, l, c = b.high, b.low, b.close
    lc = np.log(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    F = pd.DataFrame(index=b.index)
    for n in (14, 50):
        rng = h.rolling(n).max() - l.rolling(n).min()
        F[f'chop_{n}'] = 100 * np.log10(tr.rolling(n).sum() / rng.replace(0, np.nan)) / np.log10(n)
    side = np.sign(c - c.ewm(span=20, adjust=False).mean())
    F['ema20_cross_rate_50'] = (side != side.shift()).astype(float).rolling(50).mean()
    r1 = lc.diff()
    r5 = lc.diff(5)
    F['var_ratio_5_250'] = r5.rolling(250).var() / (5 * r1.rolling(250).var())
    for n in (50, 150):
        t = np.arange(n, dtype=float)
        tc_ = t - t.mean()
        cov = lc.rolling(n).apply(lambda y: np.dot(tc_, y - y.mean()), raw=True) / n
        r = cov / (tc_.std() * lc.rolling(n).std(ddof=0))
        F[f'trend_r2_{n}'] = r ** 2
    return F.astype(np.float32)


def _one(args):
    try:
        return _one_(args)
    except ZeroDivisionError:   # zero-range bars (bad data) break the SuperTrend kernel -> ticker skipped
        print('skipped (bad data)', args[0], flush=True)
        return None


def _one_(args):
    tk, g, m = args
    fn = f'{CD}/{tk}.pkl'
    if os.path.exists(fn):
        return tk
    g = g.set_index('date')[['open', 'high', 'low', 'close', 'volume']].sort_index().astype(float)
    g = g[(g[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    if len(g) < 1500:
        return None
    base, X = build(g, '1D')
    X = pd.concat([X, build_bb(g, '1D').reindex(X.index)], axis=1)
    mk = m.reindex(X.index)
    X = pd.concat([X, mk], axis=1)
    c = base.close
    for n in (63, 252):
        X[f'rel_mom_{n}'] = np.log(c / c.shift(n)) - mk[f'mkt_mom_{n if n != 252 else 252}']
    X = pd.concat([X, chop_features(base)], axis=1)
    lab, fin = oracle_labels(c.values, 1.0, return_final=True)
    f = pd.concat([base[['open', 'high', 'low', 'close']], X.astype(np.float32)], axis=1)
    f['y'] = lab
    f['final'] = np.arange(len(f)) <= fin
    f['asset'] = tk
    f['tf'] = '1D'
    f.to_pickle(fn)
    return tk


def build_all():
    os.makedirs(CD, exist_ok=True)
    d = load_all()
    m = market(d)
    jobs = [(tk, g, m) for tk, g in d.groupby('ticker')]
    with Pool(4) as p:
        done = [x for x in p.imap_unordered(_one, jobs, chunksize=4) if x]
    print('built', len(done), flush=True)


def fr(tk):
    return pd.read_pickle(f'{CD}/{tk}.pkl')


def splits():
    have = sorted(f[:-4] for f in os.listdir(CD))
    A = [t for t in STOCKS if t in have]
    rest = [t for t in have if t not in set(STOCKS)]
    rng = np.random.default_rng(0)
    rest = list(rng.permutation(rest))
    return dict(A=A, fit=rest[:200], val=rest[200:300], B=rest[300:])


def rows_before(f, cutoff, emb=24, min_len=1000):
    past = f[f.index < cutoff]
    if len(past) < min_len:
        return past.iloc[:0]
    lab, fin = oracle_labels(past.close.values, 1.0, return_final=True)
    keep = max(fin - emb, 0)
    t = past.iloc[:keep].copy()
    t['y'] = lab[:keep]
    return t


def gather(tks, cutoff):
    return pd.concat([rows_before(fr(t), cutoff) for t in tks])


def cols_of(df):
    return [c for c in df.columns if c not in NON_FEAT]


def lgb_params(p, seed=0):
    return dict(objective='binary', verbosity=-1, num_threads=4, seed=seed, bagging_freq=1,
                **{k: v for k, v in p.items() if k not in ('n_estimators', 'smooth_span', 'hysteresis')})


def window(tks, lo, hi):
    out = {}
    for t in tks:
        f = fr(t)
        w = f[(f.index >= lo) & (f.index < hi)]
        if len(w) > 250:
            out[t] = w
    return out


def per_ticker_mcc(W, pred_fn, span, h):
    ms = []
    for t, w in W.items():
        s = smooth_state(pred_fn(w), span, h)
        ok = w.final.values
        if ok.sum() > 250 and len(set(w.y.values[ok])) == 2:
            ms.append(matthews_corrcoef(w.y.values[ok], s[ok]))
    return np.median(ms)


def tune():
    S = splits()
    tr = gather(S['fit'], TUNE_CUT)
    cols = cols_of(tr)
    V = window(S['val'], TUNE_CUT, DEV_END)
    print('fit rows', tr.shape, 'val tickers', len(V), 'features', len(cols), flush=True)

    def objective(trial):
        prm = dict(learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                   num_leaves=trial.suggest_int('num_leaves', 7, 63, log=True),
                   min_child_samples=trial.suggest_int('min_child_samples', 200, 5000, log=True),
                   feature_fraction=trial.suggest_float('feature_fraction', 0.3, 0.9),
                   bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 0.9),
                   lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 100, log=True),
                   max_depth=trial.suggest_int('max_depth', 3, 8))
        n = trial.suggest_int('n_estimators', 100, 1500, log=True)
        span = trial.suggest_int('smooth_span', 1, 20, log=True)
        h = trial.suggest_float('hysteresis', 0.0, 0.35)
        m = lgb.train(lgb_params(prm, trial.number), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), n)
        v = per_ticker_mcc(V, lambda w: m.predict(w[cols]), span, h)
        print('trial', trial.number, round(v, 4), flush=True)
        return v

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=40, show_progress_bar=False)
    # baselines tuned on the same validation data
    base = {}
    for bn, grid in BASELINE_GRID.items():
        sc = []
        for prm in grid:
            ms = []
            for t, w in V.items():
                f = fr(t)
                s = np.asarray(BASELINE_FN[bn](f, prm))[f.index.get_indexer(w.index)]
                ok = w.final.values
                if ok.sum() > 250 and len(set(w.y.values[ok])) == 2:
                    ms.append(matthews_corrcoef(w.y.values[ok], s[ok]))
            sc.append((np.median(ms), prm))
        best = max(sc, key=lambda x: x[0])
        base[bn] = dict(params=best[1], val_median_mcc=best[0])
    out = dict(params=st.best_params, val_median_mcc=st.best_value, cols=cols, baselines=base,
               frozen_at=pd.Timestamp.utcnow().isoformat())
    json.dump(out, open('FROZEN_stage10.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'cols'}, indent=1, default=float), flush=True)


def final():
    fz = json.load(open('FROZEN_stage10.json'))
    S = splits()
    p, cols = fz['params'], fz['cols']
    tr = gather(S['fit'] + S['val'], DEV_END)
    m = lgb.train(lgb_params(p), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
    m.save_model('trend_model_stocks.txt')
    mc = lgb.Booster(model_file='trend_model_universal_C.txt')
    ccols = json.load(open('FROZEN_v3.json'))['C']['cols']
    rows = []
    for setname in ('A', 'B'):
        for t in S[setname]:
            f = fr(t)
            te = (f.index >= DEV_END) & f.final.values
            if te.sum() < 500:
                continue
            S_ = {'MODEL (stocks)': smooth_state(m.predict(f[cols]), p['smooth_span'], p['hysteresis']),
                  'Crypto model C': smooth_state(mc.predict(f[ccols]), 1, 0.204),
                  'EMA200': price_ema(f, 200)}
            for bn, b in fz['baselines'].items():
                S_[f'{bn} (tuned)'] = np.asarray(BASELINE_FN[bn](f, b['params']))
            for k, s in S_.items():
                for per, msk in (('2011-2026', te), ('2011-2018', te & (f.index < '2019-01-01')),
                                 ('2019-2026', te & (f.index >= '2019-01-01'))):
                    if msk.sum() < 250 or len(set(f.y.values[msk])) < 2:
                        continue
                    mt = metrics(np.asarray(s)[msk], f.y.values[msk], f.close.values[msk], fee=0.0005, bars_per_year=252)
                    mt.update(set=setname, ticker=t, clf=k, period=per)
                    rows.append(mt)
        print('set', setname, 'done', flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage10.pkl')


if __name__ == '__main__':
    {'build': build_all, 'tune': tune, 'final': final}[sys.argv[1]]()

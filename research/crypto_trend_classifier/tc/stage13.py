"""Stage 13: label v2 on 4h (see PREREG_stage13_label_v2_4h.md).  python -m tc.stage13 build|tune|final"""
import json
import os
import sys
import numpy as np
import pandas as pd
import numba
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from .dev3 import frame3
from .stage3 import frame_generic, load as oanda_load, FX, IDX
from .dataset import load_1h
from .labels import oracle_labels
from .label_v2 import label_v2
from .stage10 import chop_features
from .s12lib import new_features, lag_metrics
from .stage12 import H_GRID, C_GRID, CUSUM_BASE, to_state, cusum_base, parse_cfg
from .evaluate import BASELINE_FN, BASELINE_GRID, smooth_state
from .features import _atr

CD = 'cache_stage13'
COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT', 'DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']
HOLDOUT = ['ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
TRAIN_COINS = [c for c in COINS if c not in HOLDOUT]
OANDA = FX + IDX + ['XAU_USD']
TRAIN_ALL = TRAIN_COINS + ['BTC_BITSTAMP'] + OANDA
INNER, DEV_END = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2021-01-01', tz='UTC')
TEST_END = pd.Timestamp('2026-09-24', tz='UTC')
NON = {'open', 'high', 'low', 'close', 'y', 'final', 'asset', 'tf'}


@numba.njit(cache=True)
def _liq(tp, v, c, lo, hi, n, bins):
    T = len(c)
    below = np.full(T, np.nan)
    poc = np.full(T, np.nan)
    for t in range(n - 1, T):
        tot = 0.0
        b = 0.0
        mn, mx = 1e300, -1e300
        for i in range(t - n + 1, t + 1):
            if v[i] > 0:
                tot += v[i]
                if tp[i] < c[t]:
                    b += v[i]
            mn = min(mn, lo[i])
            mx = max(mx, hi[i])
        if tot > 0:
            below[t] = b / tot
            if mx > mn:
                h = np.zeros(bins)
                for i in range(t - n + 1, t + 1):
                    k = int((tp[i] - mn) / (mx - mn) * bins)
                    k = min(max(k, 0), bins - 1)
                    h[k] += v[i]
                kk = np.argmax(h)
                poc[t] = mn + (kk + 0.5) * (mx - mn) / bins
    return below, poc


def liquidity_features(b):
    tp = ((b.high + b.low + b.close) / 3).values
    v = np.nan_to_num(b.volume.values.astype(float))
    c, lo, hi = b.close.values, b.low.values, b.high.values
    atr = _atr(b, 14).values
    F = pd.DataFrame(index=b.index)
    for n in (30, 120):
        below, poc = _liq(tp, v, c, lo, hi, n, 20)
        F[f'liq_below_{n}'] = below
        vw = pd.Series(tp * v).rolling(n).sum().values / np.where(pd.Series(v).rolling(n).sum().values > 0,
                                                                  pd.Series(v).rolling(n).sum().values, np.nan)
        F[f'vwap_dist_{n}'] = (c - vw) / atr
        if n == 120:
            F['poc_dist_120'] = (c - poc) / atr
    return F.astype(np.float32)


def raw4h(name):
    d = load_1h(name) if (name in COINS or name == 'BTC_BITSTAMP') else oanda_load(name)
    return d.resample('4h', origin='epoch').agg(dict(open='first', high='max', low='min', close='last',
                                                     volume='sum')).dropna(subset=['close'])


def build_one(name):
    fn = f'{CD}/{name}.pkl'
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    f = frame3(name, '4h') if (name in COINS or name == 'BTC_BITSTAMP') else frame_generic(name, '4h')
    g = raw4h(name).reindex(f.index)
    if name == 'BTC_BITSTAMP':
        g['volume'] = np.nan
    X = pd.concat([new_features(g.open, g.high, g.low, g.close, g.volume),
                   chop_features(g[['high', 'low', 'close']]), liquidity_features(g)], axis=1)
    f = pd.concat([f, X], axis=1)
    f = f.loc[:, ~f.columns.duplicated()]
    f['vol_raw'] = g.volume.values
    f.to_pickle(fn)
    return f


def build():
    os.makedirs(CD, exist_ok=True)
    for n in COINS + ['BTC_BITSTAMP'] + OANDA:
        f = build_one(n)
        print(n, f.shape, f.index[0].date(), f.index[-1].date(), flush=True)


def fr(n):
    return pd.read_pickle(f'{CD}/{n}.pkl')


def rows_before(f, cutoff, emb=24):
    past = f[f.index < cutoff]
    if len(past) < 1500:
        return None
    c = past.close.values
    _, f1 = oracle_labels(c, 1.0, return_final=True)
    _, f2 = oracle_labels(c, 0.3, return_final=True)
    keep = max(min(f1, f2) - 10 - emb, 0)
    L = label_v2(past.open.values, past.high.values, past.low.values, c, past.vol_raw.values)
    t = past.iloc[:keep].copy()
    t['yv2'] = L.y.values[:keep]
    return t


def gather(names, cutoff):
    return pd.concat([x for x in (rows_before(fr(n), cutoff) for n in names) if x is not None])


def feat_cols():
    fz = json.load(open('FROZEN_v3.json'))['C']['cols']
    extra = [c for c in fr('BTCUSDT').columns if c.startswith(('nret_', 'macdn_', 'cusum_', 'bocpd_', 'obv_', 'ad_',
                                                               'chop_', 'ema20_cross', 'var_ratio', 'trend_r2',
                                                               'liq_', 'vwap_', 'poc_'))]
    return list(dict.fromkeys(fz + extra))


def params(p, seed=0):
    return dict(objective='regression', verbosity=-1, num_threads=4, seed=seed, bagging_freq=1,
                **{k: v for k, v in p.items() if k != 'n_estimators'})


def score(m, X, sd):
    return np.tanh(m.predict(X) / sd)


def eval_window(state, f, lo, hi):
    w = (f.index >= lo) & (f.index < hi) & f.final.values
    if w.sum() < 100 or len(set(f.y.values[w])) < 2:
        return None
    return lag_metrics(np.asarray(state, np.int8)[w], f.y.values[w].astype(np.int8), np.log(f.close.values[w]))


def pick(cands, fn_states, frames, lo, hi):
    """cands: list of configs; fn_states(cfg, name, f) -> state. Returns table (cfg, fpt, miss, delay)."""
    out = []
    for cfg in cands:
        fp, mi, dl = [], [], []
        for n, f in frames.items():
            m = eval_window(fn_states(cfg, n, f), f, lo, hi)
            if m is not None and m[4] > 0:
                fp.append(m[0]); mi.append(m[1]); dl.append(m[2])
        out.append(dict(cfg=str(cfg), fpt=np.median(fp), miss=np.median(mi), delay=np.median(dl)))
    return pd.DataFrame(out)


def best_under_budget(T):
    ok = T[T.fpt <= 2.0]
    return ok.sort_values('miss').iloc[0] if len(ok) else T.sort_values('fpt').iloc[0]


def tune():
    cols = feat_cols()
    tr = gather(TRAIN_ALL, INNER)
    vf = {n: fr(n) for n in TRAIN_COINS}
    va = pd.concat([x[(x.index >= INNER)] for x in (rows_before(f, DEV_END) for f in vf.values()) if x is not None])
    print('train', tr.shape, 'valid', va.shape, 'features', len(cols), flush=True)

    def objective(trial):
        p = dict(learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int('num_leaves', 7, 63, log=True),
                 min_child_samples=trial.suggest_int('min_child_samples', 200, 5000, log=True),
                 feature_fraction=trial.suggest_float('feature_fraction', 0.3, 0.9),
                 bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 0.9),
                 lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 100, log=True),
                 max_depth=trial.suggest_int('max_depth', 3, 8),
                 n_estimators=trial.suggest_int('n_estimators', 100, 1500, log=True))
        m = lgb.train(params(p, trial.number), lgb.Dataset(tr[cols], tr.yv2), p['n_estimators'])
        rho = spearmanr(m.predict(va[cols]), va.yv2).correlation
        print('trial', trial.number, round(rho, 4), flush=True)
        return rho

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=30, show_progress_bar=False)
    p = st.best_params
    m = lgb.train(params(p), lgb.Dataset(tr[cols], tr.yv2), p['n_estimators'])
    sd = float(np.std(m.predict(tr[cols].iloc[::5])))
    sc = {n: score(m, f[cols], sd) for n, f in vf.items()}
    T = pick(H_GRID + C_GRID, lambda cfg, n, f: to_state(sc[n], cfg), vf, INNER, DEV_END)
    bm = best_under_budget(T)
    base = {}
    for name, grid in list(BASELINE_GRID.items()) + [('CUSUM', CUSUM_BASE)]:
        fn = (lambda cfg, n, f: cusum_base(f.close.values, **cfg)) if name == 'CUSUM' else \
             (lambda cfg, n, f, name=name: np.asarray(BASELINE_FN[name](f, cfg)))
        Tb = pick(grid, fn, vf, INNER, DEV_END)
        b = best_under_budget(Tb)
        base[name] = dict(cfg=b.cfg, val_miss=b.miss, val_fpt=b.fpt, val_delay=b.delay)
    out = dict(params=p, val_spearman=st.best_value, sd=sd, conversion=dict(cfg=bm.cfg, val_miss=bm.miss, val_fpt=bm.fpt,
               val_delay=bm.delay), baselines=base, cols=cols, frozen_at=str(pd.Timestamp.now('UTC')))
    json.dump(out, open('FROZEN_stage13.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'cols'}, indent=1, default=float), flush=True)


if __name__ == '__main__':
    {'build': build, 'tune': tune}.get(sys.argv[1], lambda: None)()


def stage12_states():
    """stage-12 stock model applied unchanged to all 10 coins at 4h (fixed conversion)."""
    from .stage11 import basket_market, frame as cframe
    from .stage12 import score as s12score, to_state as s12state
    meta = json.load(open('stage12_model_meta.json'))
    m12 = lgb.Booster(model_file='trend_model_stage12.txt')
    c12 = json.load(open('FROZEN_stage12.json'))['cols']
    mk = basket_market('4h')
    out = {}
    for a in COINS:
        f = cframe(a, '4h', mk)
        g = raw4h(a).reindex(f.index)
        f = pd.concat([f, new_features(g.open, g.high, g.low, g.close, g.volume)], axis=1)
        out[a] = pd.Series(s12state(s12score(m12, meta['kind'], meta['sd'], f[c12]), parse_cfg(meta['cfg'])), f.index)
    return out


def final():
    FZ = json.load(open('FROZEN_stage13.json'))
    p, cols, sd = FZ['params'], FZ['cols'], FZ['sd']
    tr = gather(TRAIN_ALL, DEV_END)
    m = lgb.train(params(p), lgb.Dataset(tr[cols], tr.yv2), p['n_estimators'])
    sd = float(np.std(m.predict(tr[cols].iloc[::5])))
    m.save_model('trend_model_stage13.txt')
    json.dump(dict(sd=sd, cfg=FZ['conversion']['cfg']), open('stage13_model_meta.json', 'w'))
    o = pd.read_pickle('out_stage2_C.pkl')
    s12 = stage12_states()
    from sklearn.metrics import matthews_corrcoef
    rows = []
    for a in COINS:
        f = fr(a)
        L = label_v2(f.open.values, f.high.values, f.low.values, f.close.values, f.vol_raw.values).y.values
        sc = score(m, f[cols], sd)
        S = {'STAGE-13 MODEL': to_state(sc, parse_cfg(FZ['conversion']['cfg']))}
        for bn, b in FZ['baselines'].items():
            cfg = parse_cfg(b['cfg'])
            S[bn] = cusum_base(f.close.values, **cfg) if bn == 'CUSUM' else np.asarray(BASELINE_FN[bn](f, cfg))
        x = o[(o.asset == a) & (o.tf == '4h')]
        cm = pd.Series(np.nan, f.index)
        ii = x.index.intersection(f.index)
        cm.loc[ii] = x.p.reindex(ii).values
        S['Crypto model C (walk-forward)'] = smooth_state(cm.values, 1, 0.204)
        S['Stage-12 model'] = s12[a].reindex(f.index).fillna(0).values.astype(np.int8)
        for yr in range(2021, 2027):
            lo, hi = pd.Timestamp(f'{yr}-01-01', tz='UTC'), min(pd.Timestamp(f'{yr + 1}-01-01', tz='UTC'), TEST_END)
            w = (f.index >= lo) & (f.index < hi) & f.final.values
            if w.sum() < 200 or len(set(f.y.values[w])) < 2:
                continue
            lc = np.log(f.close.values[w])
            r = np.r_[np.diff(lc), 0.0]
            for k, s in S.items():
                s = np.asarray(s, np.int8)
                fpt, mis, dl, gv, nseg = lag_metrics(s[w], f.y.values[w].astype(np.int8), lc)
                pos = (s[w] == 1).astype(float)
                pnl = pos * r - 0.001 * np.abs(np.diff(pos, prepend=0.0))
                rows.append(dict(group='held-out coin' if a in HOLDOUT else 'training coin', asset=a, year=yr, clf=k,
                                 fpt=fpt, miss=mis, delay=dl, giveback=gv, nseg=nseg,
                                 mcc=matthews_corrcoef(f.y.values[w], np.where(s[w] == 1, 1, -1)),
                                 rho_label=spearmanr(sc[w], L[w]).correlation if k == 'STAGE-13 MODEL' else np.nan,
                                 lf_sharpe=pnl.mean() / pnl.std() * np.sqrt(2190) if pnl.std() > 0 else np.nan,
                                 bh_sharpe=r.mean() / r.std() * np.sqrt(2190)))
        print(a, flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage13_test.pkl')
    pd.set_option('display.width', 220)
    print(R.groupby(['group', 'clf'])[['miss', 'delay', 'fpt', 'giveback', 'mcc', 'rho_label', 'lf_sharpe', 'bh_sharpe']]
          .median().round(3).to_string())


if __name__ == '__main__' and sys.argv[1] == 'final':
    final()

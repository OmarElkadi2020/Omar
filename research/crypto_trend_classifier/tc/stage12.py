"""Stage 12: faster trend detection (see PREREG_stage12_fast_trend.md).  python -m tc.stage12 build|cpcv|final"""
import json
import os
import sys
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from .labels import oracle_labels
from .stage10 import splits, lgb_params, NON_FEAT, CD as CD10
from .s12lib import (new_features, ternary_labels, trend_scan, remaining_value, conv_h, conv_c, ewm_score,
                     lag_metrics)
from .evaluate import BASELINE_FN, BASELINE_GRID

CD = 'cache_stage12'
DEV0, DEV1 = pd.Timestamp('1996-01-01', tz='UTC'), pd.Timestamp('2011-01-01', tz='UTC')
LABELS = ['B0.5', 'B1', 'B2', 'B1w', 'T1', 'TS', 'RV']
LBCOLS = {'lb_B0.5', 'lb_B1', 'lb_B2', 'lb_T1', 'lb_TS', 'w_TS', 'lb_RV', 'w_B1w'}
H_GRID = [('H', sp, ti, to) for sp in (1, 3, 5, 10) for ti in (0.1, 0.2, 0.3, 0.4, 0.5)
          for to in (-ti, 0.0, ti / 2)]
C_GRID = [('C', 0, ka, h) for ka in (0.0, 0.1, 0.2, 0.3) for h in (0.5, 1, 2, 4, 8)]
CUSUM_BASE = [dict(kappa=k, h=h) for k in (0.1, 0.25, 0.5, 1.0) for h in (2, 4, 8, 16, 32)]


def dev_labels(f):
    """training labels computed on the series truncated at DEV1 (finality respected)."""
    past = f[f.index < DEV1]
    out = pd.DataFrame(index=f.index, dtype=float)
    for c in LBCOLS:
        out[c] = np.nan
    if len(past) < 600:
        return out
    c = past.close.values
    n = len(past)
    for k, name in ((0.5, 'lb_B0.5'), (1.0, 'lb_B1'), (2.0, 'lb_B2')):
        lab, fin = oracle_labels(c, k, return_final=True)
        keep = max(fin - 24, 0)
        v = np.full(n, np.nan)
        v[:keep] = lab[:keep]
        out.loc[past.index, name] = v
        if k == 1.0:
            w = np.ones(n)
            ch = np.flatnonzero(np.diff(lab)) + 1
            for a, b in zip(np.r_[0, ch], np.r_[ch, n]):
                w[a:a + max(1, int(0.2 * (b - a)))] = 3.0
            out.loc[past.index, 'w_B1w'] = w
            rv = remaining_value(c, lab, keep)
            out.loc[past.index, 'lb_RV'] = rv
    lab, fin = ternary_labels(c, 1.0)
    keep = max(fin - 24, 0)
    v = np.full(n, np.nan)
    v[:keep] = lab[:keep]
    out.loc[past.index, 'lb_T1'] = v
    tv = trend_scan(np.log(c), 5, 60)
    tv[n - 61:] = np.nan                      # forward window must end before the cut-off
    out.loc[past.index, 'lb_TS'] = np.sign(tv)
    out.loc[past.index, 'w_TS'] = np.minimum(np.abs(tv), 30.0)
    return out


def _one(tk):
    fn = f'{CD}/{tk}.pkl'
    if os.path.exists(fn):
        return tk
    f = pd.read_pickle(f'{CD10}/{tk}.pkl')
    g = RAW.get_group(tk).set_index('date').sort_index()[['open', 'high', 'low', 'close', 'volume']].astype(float)
    g = g[~g.index.duplicated()].reindex(f.index)
    F = new_features(g.open, g.high, g.low, g.close, g.volume)
    f = pd.concat([f, F, dev_labels(f)], axis=1)
    f.to_pickle(fn)
    return tk


def build():
    global RAW
    os.makedirs(CD, exist_ok=True)
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    d['date'] = pd.to_datetime(d.date).dt.tz_localize('UTC')
    RAW = d.groupby('ticker')
    tks = sorted(x[:-4] for x in os.listdir(CD10))
    with Pool(4) as p:
        n = sum(1 for _ in p.imap_unordered(_one, tks, chunksize=4))
    print('built', n, flush=True)


def fr(tk):
    return pd.read_pickle(f'{CD}/{tk}.pkl')


def feat_cols(f):
    return [c for c in f.columns if c not in NON_FEAT and c not in LBCOLS]


def target(df, lab):
    """(X-rows mask, y, weight, kind) for a label name."""
    if lab in ('B0.5', 'B1', 'B2', 'B1w'):
        col = 'lb_' + ('B1' if lab == 'B1w' else lab)
        m = df[col].notna()
        y = (df.loc[m, col] == 1).astype(int)
        w = df.loc[m, 'w_B1w'].values if lab == 'B1w' else None
        return m, y, w, 'bin'
    if lab == 'T1':
        m = df['lb_T1'].notna()
        return m, (df.loc[m, 'lb_T1'] + 1).astype(int), None, 'multi'
    if lab == 'TS':
        m = df['lb_TS'].notna() & (df['lb_TS'] != 0)
        return m, (df.loc[m, 'lb_TS'] == 1).astype(int), df.loc[m, 'w_TS'].values, 'bin'
    if lab == 'RV':
        m = df['lb_RV'].notna()
        return m, df.loc[m, 'lb_RV'].clip(-10, 10), None, 'reg'


def train(df, lab, cols, p):
    m, y, w, kind = target(df, lab)
    prm = lgb_params(p)
    if kind == 'multi':
        prm.update(objective='multiclass', num_class=3)
    if kind == 'reg':
        prm.update(objective='regression')
    mdl = lgb.train(prm, lgb.Dataset(df.loc[m, cols], y, weight=w), p['n_estimators'])
    sd = float(np.std(mdl.predict(df.loc[m, cols].iloc[::7]))) if kind == 'reg' else 1.0
    return mdl, kind, sd


def score(mdl, kind, sd, X):
    p = mdl.predict(X)
    if kind == 'bin':
        return 2 * p - 1
    if kind == 'multi':
        return p[:, 2] - p[:, 0]
    return np.tanh(p / sd)


def to_state(s, cfg):
    kind, a, b, c = cfg
    if kind == 'H':
        return conv_h(ewm_score(s, a), b, c)
    return conv_c(np.asarray(s, float), b, c)


def cusum_base(close, kappa, h):
    lc = np.log(close)
    r = np.r_[0.0, np.diff(lc)]
    from .labels import ewm_mean
    sig = np.sqrt(ewm_mean(r * r, 20))
    z = np.r_[0.0, r[1:] / np.maximum(sig[:-1], 1e-9)]
    return conv_c(np.clip(z, -8, 8), kappa, h)


def blocks():
    edges = pd.date_range(DEV0, DEV1, periods=7)
    return [(edges[i], edges[i + 1]) for i in range(6)]


_MASK = {}


def eval_state(state, f, lo, hi):
    key = (id(f), lo)
    if key not in _MASK:
        w = (f.index >= lo) & (f.index < hi) & f.final.values
        _MASK[key] = (w, f.y.values[w].astype(np.int8), np.log(f.close.values[w])) if w.sum() >= 100 else None
    e = _MASK[key]
    if e is None:
        return None
    w, y, lc = e
    return lag_metrics(np.asarray(state, np.int8)[w], y, lc)


def cpcv():
    fz = json.load(open('FROZEN_stage10.json'))
    p = fz['params']
    S = splits()
    B = blocks()
    fit = {t: fr(t) for t in S['fit']}
    val = {t: fr(t) for t in S['val']}
    cols = feat_cols(next(iter(fit.values())))
    FIT = pd.concat([f[f.index < DEV1].assign(_tk=t) for t, f in fit.items()])
    print('features', len(cols), 'fit rows', len(FIT), flush=True)
    combos = list(itertools.combinations(range(6), 2))
    res = []            # (candidate, split, fpt_median, miss_median, delay_median)
    for si, (i, j) in enumerate(combos):
        emb = pd.Timedelta(days=31)
        keep = np.ones(len(FIT), bool)
        for b in (i, j):
            keep &= ~((FIT.index >= B[b][0] - emb) & (FIT.index < B[b][1] + emb))
        tr = FIT[keep]
        for lab in LABELS:
            mdl, kind, sd = train(tr, lab, cols, p)
            sc = {t: score(mdl, kind, sd, f[cols]) for t, f in val.items()}
            for cfg in H_GRID + C_GRID:
                fpt, mis, dl = [], [], []
                for t, f in val.items():
                    st = to_state(sc[t], cfg)
                    for b in (i, j):
                        m = eval_state(st, f, *B[b])
                        if m is not None and m[4] > 0:
                            fpt.append(m[0]); mis.append(m[1]); dl.append(m[2])
                res.append(dict(cand=f'{lab}|{cfg}', label=lab, cfg=str(cfg), split=si,
                                fpt=np.median(fpt), miss=np.median(mis), delay=np.median(dl)))
        print('split', si, (i, j), flush=True)
    # baselines (no training): same splits and blocks
    base = [('EMA cross', prm) for prm in BASELINE_GRID['EMA cross']] + \
           [('Price vs EMA', prm) for prm in BASELINE_GRID['Price vs EMA']] + \
           [('SuperTrend', prm) for prm in BASELINE_GRID['SuperTrend']] + \
           [('Online directional-change', prm) for prm in BASELINE_GRID['Online directional-change']] + \
           [('CUSUM', prm) for prm in CUSUM_BASE]
    calm = lgb.Booster(model_file='trend_model_stocks.txt')
    fzb = json.load(open('FROZEN_stage10b.json'))
    from .evaluate import smooth_state
    for name, prm in base + [('Stage-10 calm model', None)]:
        sts = {}
        for t, f in val.items():
            if name == 'CUSUM':
                sts[t] = cusum_base(f.close.values, **prm)
            elif name == 'Stage-10 calm model':
                sts[t] = smooth_state(calm.predict(f[fz['cols']]), fzb['span'], fzb['h'])
            else:
                sts[t] = np.asarray(BASELINE_FN[name](f, prm))
        for si, (i, j) in enumerate(combos):
            fpt, mis, dl = [], [], []
            for t, f in val.items():
                for b in (i, j):
                    m = eval_state(sts[t], f, *B[b])
                    if m is not None and m[4] > 0:
                        fpt.append(m[0]); mis.append(m[1]); dl.append(m[2])
            res.append(dict(cand=f'BASE {name}|{prm}', label='BASE ' + name, cfg=str(prm), split=si,
                            fpt=np.median(fpt), miss=np.median(mis), delay=np.median(dl)))
    R = pd.DataFrame(res)
    R.to_pickle('out_stage12_cpcv.pkl')
    A = R.groupby(['cand', 'label', 'cfg'])[['fpt', 'miss', 'delay']].mean().reset_index()
    ok = A[A.fpt <= 2.0]
    best = ok.sort_values('miss').groupby('label').head(1).sort_values('miss')
    pd.set_option('display.width', 220)
    print(best.round(3).to_string(index=False), flush=True)
    model_best = best[~best.label.str.startswith('BASE')].iloc[0]
    base_best = best[best.label.str.startswith('BASE')].iloc[0]
    frozen = dict(model=dict(label=model_best.label, cfg=model_best.cfg, cpcv_miss=model_best.miss,
                             cpcv_fpt=model_best.fpt, cpcv_delay=model_best.delay),
                  baselines={r.label: dict(cfg=r.cfg, cpcv_miss=r.miss, cpcv_fpt=r.fpt, cpcv_delay=r.delay)
                             for r in best[best.label.str.startswith('BASE')].itertuples()},
                  best_baseline=base_best.label, cols=cols, frozen_at=pd.Timestamp.utcnow().isoformat())
    json.dump(frozen, open('FROZEN_stage12.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    {'build': build, 'cpcv': cpcv}.get(sys.argv[1], lambda: None)()


# ------------------------------------------------------------------ one-shot test
def parse_cfg(s):
    import ast
    return ast.literal_eval(s)


def base_state(name, cfg, f):
    prm = parse_cfg(cfg)
    if name == 'CUSUM':
        return cusum_base(f.close.values, **prm)
    return np.asarray(BASELINE_FN[name](f, prm))


def crypto_frames():
    from .stage11 import basket_market, frame as cframe
    from .dataset import load_1h
    out = {}
    for tf in ('1D', '4h'):
        m = basket_market(tf)
        for a in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']:
            f = cframe(a, tf, m)
            d = load_1h(a)
            g = d.resample(tf, origin='epoch').agg(dict(open='first', high='max', low='min', close='last', volume='sum'))
            g = g.reindex(f.index)
            f = pd.concat([f, new_features(g.open, g.high, g.low, g.close, g.volume)], axis=1)
            out[(a, tf)] = f
    return out


def final():
    fz10 = json.load(open('FROZEN_stage10.json'))
    fzb = json.load(open('FROZEN_stage10b.json'))
    FZ = json.load(open('FROZEN_stage12.json'))
    p, cols = fz10['params'], FZ['cols']
    S = splits()
    TR = pd.concat([fr(t)[lambda x: x.index < DEV1] for t in S['fit'] + S['val']])
    lab, cfg = FZ['model']['label'], parse_cfg(FZ['model']['cfg'])
    mdl, kind, sd = train(TR, lab, cols, p)
    mdl.save_model('trend_model_stage12.txt')
    json.dump(dict(kind=kind, sd=sd, label=lab, cfg=FZ['model']['cfg']), open('stage12_model_meta.json', 'w'))
    calm = lgb.Booster(model_file='trend_model_stocks.txt')
    from .evaluate import smooth_state, metrics
    rows = []

    def evaluate(setname, name, f, lo, hi, bpy):
        sc = score(mdl, kind, sd, f[cols])
        states = {'STAGE-12 MODEL': to_state(sc, cfg),
                  'Stage-10 calm model': smooth_state(calm.predict(f[fz10['cols']]), fzb['span'], fzb['h'])}
        for bl, b in FZ['baselines'].items():
            if bl == 'BASE Stage-10 calm model':
                continue                     # already included above
            states[bl.replace('BASE ', '')] = base_state(bl.replace('BASE ', ''), b['cfg'], f)
        w = (f.index >= lo) & (f.index < hi) & f.final.values
        if w.sum() < 200 or len(set(f.y.values[w])) < 2:
            return
        lc = np.log(f.close.values[w])
        for k, s in states.items():
            s = np.asarray(s, np.int8)
            fpt, mis, dl, gv, nseg = lag_metrics(s[w], f.y.values[w].astype(np.int8), lc)
            pos = (s[w] == 1).astype(float)
            r = np.r_[np.diff(lc), 0.0]
            pnl = pos * r - 0.001 * np.abs(np.diff(pos, prepend=0.0))
            from sklearn.metrics import matthews_corrcoef
            rows.append(dict(set=setname, asset=name, clf=k, fpt=fpt, miss=mis, delay=dl, giveback=gv, nseg=nseg,
                             mcc=matthews_corrcoef(f.y.values[w], np.where(s[w] == 1, 1, -1)),
                             lf_sharpe=pnl.mean() / pnl.std() * np.sqrt(bpy) if pnl.std() > 0 else np.nan,
                             bh_sharpe=r.mean() / r.std() * np.sqrt(bpy)))

    end = pd.Timestamp('2027-01-01', tz='UTC')
    for setname in ('A', 'B'):
        for t in S[setname]:
            evaluate(setname, t, fr(t), DEV1, end, 252)
        print('set', setname, 'done', flush=True)
    for (a, tf), f in crypto_frames().items():
        evaluate(f'Crypto {tf}', a, f, pd.Timestamp('2019-01-01', tz='UTC'), end, {'1D': 365, '4h': 2190}[tf])
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage12_test.pkl')
    pd.set_option('display.width', 220)
    print(R.groupby(['set', 'clf'])[['miss', 'delay', 'fpt', 'giveback', 'mcc', 'lf_sharpe', 'bh_sharpe']].median().round(3).to_string())


if __name__ == '__main__' and sys.argv[1] == 'final':
    final()

"""Quick test: do cross-sectional features (PCA absorption, avg correlation, dispersion, rank, systematic vs
idiosyncratic move) improve the frozen trend classifier C? Same params, same folds, only the columns differ."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dev3 import frame3
from .dataset import COINS
from .tune import gather
from .stage2 import params
from .evaluate import smooth_state, metrics

EVAL = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT']
XS = ['xs_ar', 'xs_ar7', 'xs_dar', 'xs_corr', 'xs_disp7', 'xs_mkt7', 'xs_mkt30', 'xs_resid7', 'xs_resid30', 'xs_rank14', 'xs_n']
_T = {}


def table(tf, suf):
    C = pd.concat({a: frame3(a + suf, tf).close for a in COINS}, axis=1).sort_index()
    R = np.log(C).diff()
    per = pd.Timedelta('1D') // pd.Timedelta(tf)
    W, S = 30 * per, 7 * per
    Rv = R.values
    ar, ar7, cr = (np.full(len(R), np.nan) for _ in range(3))
    for t in range(W, len(R)):
        for win, out in ((W, ar), (S, ar7)):
            X = Rv[t - win + 1:t + 1]
            ok = ~np.isnan(X).any(0) & (np.nanstd(X, 0) > 0)
            if ok.sum() >= 3:
                c = np.corrcoef(X[:, ok], rowvar=False)
                ev = np.linalg.eigvalsh(c)
                out[t] = ev[-1] / ev.sum()
                if win == W:
                    k = c.shape[0]
                    cr[t] = (c.sum() - k) / (k * (k - 1))
    T = pd.DataFrame(index=R.index)
    T['xs_ar'], T['xs_ar7'], T['xs_corr'] = ar, ar7, cr
    T['xs_dar'] = T.xs_ar7 - T.xs_ar
    r7, r30, r14 = np.log(C / C.shift(S)), np.log(C / C.shift(W)), np.log(C / C.shift(14 * per))
    T['xs_disp7'] = r7.std(axis=1)
    T['xs_mkt7'], T['xs_mkt30'] = r7.mean(axis=1), r30.mean(axis=1)
    T['xs_n'] = C.notna().sum(axis=1)
    return T, r7, r30, r14.rank(axis=1, pct=True)


def xs_join(fr):
    name, tf = fr.asset.iloc[0], fr.tf.iloc[0]
    suf = '_INV' if name.endswith('_INV') else ''
    if (tf, suf) not in _T:
        _T[(tf, suf)] = table(tf, suf)
    T, r7, r30, rk = _T[(tf, suf)]
    x = T.reindex(fr.index)
    base = name.replace('_INV', '')
    if base in COINS:
        x['xs_resid7'] = (r7[base] - T.xs_mkt7).reindex(fr.index)
        x['xs_resid30'] = (r30[base] - T.xs_mkt30).reindex(fr.index)
        x['xs_rank14'] = rk[base].reindex(fr.index)
    else:
        x['xs_resid7'] = x['xs_resid30'] = x['xs_rank14'] = np.nan
    return pd.concat([fr, x[XS].astype(np.float32)], axis=1)


def fr_fn(a, tf):
    return xs_join(frame3(a, tf))


def main():
    fz = json.load(open('FROZEN_v3.json'))['C']
    p = fz['params']
    models = {'C (frozen)': fz['cols'], 'C + cross-section': fz['cols'] + XS}
    cuts = ['2019-01-01', '2021-01-01', '2023-01-01', '2025-01-01', '2026-09-24']
    rows, imp = [], {}
    for i in range(4):
        tr = gather(cuts[i], fr_fn=fr_fn)
        lo, hi = pd.Timestamp(cuts[i], tz='UTC'), pd.Timestamp(cuts[i + 1], tz='UTC')
        assert tr.index.max() < lo
        for mn, cols in models.items():
            m = lgb.train(params(p), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
            if mn != 'C (frozen)':
                g = pd.Series(m.feature_importance('gain'), cols)
                imp[cuts[i]] = (g[XS] / g.sum()).round(4).to_dict()
            for a in EVAL:
                for tf in ('1h', '4h'):
                    fr = fr_fn(a, tf)
                    te = fr[(fr.index >= lo) & (fr.index < hi)]
                    s = smooth_state(m.predict(te[cols]), p['smooth_span'], p['hysteresis'])
                    ok = te.final.values
                    mt = metrics(s[ok], te.y.values[ok], te.close.values[ok])
                    mt.update(model=mn, asset=a, tf=tf, fold=cuts[i])
                    rows.append(mt)
        print('fold', cuts[i], flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_xs.pkl')
    json.dump(imp, open('out_xs_importance.json', 'w'), indent=1)
    k = ['mcc', 'flip_precision', 'flips_per_oracle_flip', 'missed_move_frac', 'capture']
    pd.set_option('display.width', 200)
    print(R.groupby(['tf', 'model'])[k].mean().round(3))
    w = R.pivot_table(index=['asset', 'tf', 'fold'], columns='model', values='mcc')
    print('cells where cross-section improves MCC:', int((w['C + cross-section'] > w['C (frozen)']).sum()), 'of', len(w))
    print(json.dumps(imp, indent=1))


if __name__ == '__main__':
    main()

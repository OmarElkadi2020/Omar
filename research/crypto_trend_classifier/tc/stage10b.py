"""Stage 10b: a CALM version of the stage-10 model (few false flips). Smoothing chosen on the 2006-2010
validation tickers only (model trained on fit tickers < 2006), constraint: median flips per true flip <= 2.0,
then applied once to the test sets."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from .stage10 import splits, gather, window, fr, lgb_params, TUNE_CUT, DEV_END
from .evaluate import smooth_state, metrics, BASELINE_FN

GRID = [(sp, h) for sp in (1, 3, 5, 10, 15, 20, 30, 40, 60) for h in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)]


def flips(s, y):
    return np.sum(s[1:] != s[:-1]) / max(np.sum(y[1:] != y[:-1]), 1)


def main():
    fz = json.load(open('FROZEN_stage10.json'))
    p, cols = fz['params'], fz['cols']
    S = splits()
    tr = gather(S['fit'], TUNE_CUT)
    m = lgb.train(lgb_params(p), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
    V = window(S['val'], TUNE_CUT, DEV_END)
    P = {t: m.predict(w[cols]) for t, w in V.items()}
    res = []
    for sp, h in GRID:
        ms, fs = [], []
        for t, w in V.items():
            ok = w.final.values
            if ok.sum() < 250 or len(set(w.y.values[ok])) < 2:
                continue
            s = smooth_state(P[t], sp, h)
            ms.append(matthews_corrcoef(w.y.values[ok], s[ok]))
            fs.append(flips(s[ok], w.y.values[ok]))
        res.append(dict(span=sp, h=h, mcc=np.median(ms), flips=np.median(fs)))
    G = pd.DataFrame(res)
    ok = G[G.flips <= 2.0]
    best = ok.sort_values('mcc').iloc[-1]
    # same constraint for the baselines: best parameter with median flips <= 2.0 on the same validation data
    bgrid = {'Price vs EMA': [dict(n=n) for n in (20, 50, 100, 150, 200, 300)],
             'EMA cross': [dict(fast=f, slow=s) for f in (5, 10, 20, 50) for s in (50, 100, 200, 300) if f < s],
             'SuperTrend': [dict(period=pp, factor=ff) for pp in (10, 24, 48) for ff in (2, 3, 4, 5, 6)],
             'Online directional-change': [dict(k=k) for k in (0.5, 1.0, 1.5, 2.0, 3.0)]}
    bsel = {}
    for bn, grid in bgrid.items():
        cand = []
        for prm in grid:
            ms, fs = [], []
            for t, w in V.items():
                f = fr(t)
                ok_ = w.final.values
                if ok_.sum() < 250 or len(set(w.y.values[ok_])) < 2:
                    continue
                s = np.asarray(BASELINE_FN[bn](f, prm))[f.index.get_indexer(w.index)]
                ms.append(matthews_corrcoef(w.y.values[ok_], s[ok_]))
                fs.append(flips(s[ok_], w.y.values[ok_]))
            cand.append((np.median(ms), np.median(fs), prm))
        c2 = [c for c in cand if c[1] <= 2.0]
        if c2:
            b = max(c2, key=lambda c: c[0])
            bsel[bn] = dict(params=b[2], val_mcc=b[0], val_flips=b[1])
    frozen = dict(span=int(best.span), h=float(best.h), val_mcc=float(best.mcc), val_flips=float(best.flips),
                  baselines=bsel, frozen_at=pd.Timestamp.utcnow().isoformat())
    json.dump(frozen, open('FROZEN_stage10b.json', 'w'), indent=1, default=float)
    print(json.dumps(frozen, indent=1, default=float), flush=True)
    # one-shot test with the final stage-10 model
    mf = lgb.Booster(model_file='trend_model_stocks.txt')
    mc = lgb.Booster(model_file='trend_model_universal_C.txt')
    ccols = json.load(open('FROZEN_v3.json'))['C']['cols']
    rows = []
    for setname in ('A', 'B'):
        for t in S[setname]:
            f = fr(t)
            te = (f.index >= DEV_END) & f.final.values
            if te.sum() < 500 or len(set(f.y.values[te])) < 2:
                continue
            SS = {'MODEL calm': smooth_state(mf.predict(f[cols]), frozen['span'], frozen['h']),
                  'Crypto model C': smooth_state(mc.predict(f[ccols]), 1, 0.204)}
            for bn, b in bsel.items():
                SS[f'{bn} (calm-tuned)'] = np.asarray(BASELINE_FN[bn](f, b['params']))
            for k, s in SS.items():
                mt = metrics(np.asarray(s)[te], f.y.values[te], f.close.values[te], fee=0.0005, bars_per_year=252)
                mt.update(set=setname, ticker=t, clf=k)
                rows.append(mt)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage10b.pkl')


if __name__ == '__main__':
    main()

"""Stage 4: tune on pre-2019 crypto only, freeze, then one test run (see PREREG_stage4.md)."""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .dev3 import frame3
from .dataset import COINS
from .tune import gather, ASSETS
from .stage2 import params
from .labels import oracle_labels
from .dips import (dip_trades, model_on, baseline_on, price_ema_on, tstat, diagnostics, GRIDS, MODEL_GRID, HOLD)

END_DEV = pd.Timestamp('2019-01-01', tz='UTC')


def dev_probs():
    fz = json.load(open('FROZEN_v3.json'))['C']
    out = {}
    for Y in (2016, 2017, 2018):
        tr = gather(f'{Y}-01-01', fr_fn=frame3)
        m = lgb.train(params(fz['params']), lgb.Dataset(tr[fz['cols']], (tr.y == 1).astype(int)), fz['params']['n_estimators'])
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        for a in ASSETS:
            for tf in ('1h', '4h'):
                fr = frame3(a, tf)
                te = fr[(fr.index >= lo) & (fr.index < hi)]
                if len(te):
                    out.setdefault((a, tf), []).append(pd.Series(m.predict(te[fz['cols']]), te.index))
        print('dev fold', Y, flush=True)
    return {k: pd.concat(v).sort_index() for k, v in out.items()}


def collect(series):
    """series: {(name, tf): (frame, prob Series or None, lo, hi)} -> per series events & R within [lo, hi)."""
    rows = {}
    for key, (fr, p, lo, hi) in series.items():
        ev, R = dip_trades(fr)
        t = fr.index[ev]
        keep = (t >= lo) & (t < hi)
        # outcome window must also end before hi (no peeking past the evaluation window)
        keep &= (ev + HOLD) < fr.index.searchsorted(hi)
        rows[key] = (fr, p, ev[keep], R[keep])
    return rows


def score_filter(rows, on_fn):
    Rs = []
    for key, (fr, p, ev, R) in rows.items():
        on = on_fn(fr, p)
        Rs.append(R[on[ev] == 1])
    Rs = np.concatenate(Rs) if Rs else np.array([])
    return tstat(Rs), len(Rs), (Rs.mean() if len(Rs) else np.nan), (np.mean(Rs > 0) if len(Rs) else np.nan)


def model_fn(cfg):
    sp, a, b = cfg

    def f(fr, p):
        full = pd.Series(np.nan, fr.index)
        full.loc[p.index] = p.values
        return model_on(full.values, sp, a, b)
    return f


def tune():
    P = dev_probs()
    series = {}
    for (a, tf), p in P.items():
        series[(a, tf)] = (frame3(a, tf), p, p.index.min(), END_DEV)
    rows = collect(series)
    res = {}
    best = max(MODEL_GRID, key=lambda c: np.nan_to_num(score_filter(rows, model_fn(c))[0], nan=-9))
    res['Model'] = dict(cfg=list(best), dev=score_filter(rows, model_fn(best)))
    for fam, grid in GRIDS.items():
        bp = max(grid, key=lambda prm: np.nan_to_num(score_filter(rows, lambda fr, p, prm=prm: baseline_on(fr, fam, prm))[0], nan=-9))
        res[fam] = dict(params=bp, dev=score_filter(rows, lambda fr, p, prm=bp: baseline_on(fr, fam, prm)))
    res['EMA200 (fixed)'] = dict(params=dict(n=200), dev=score_filter(rows, lambda fr, p: price_ema_on(fr, 200)))
    res['EMA100 (fixed)'] = dict(params=dict(n=100), dev=score_filter(rows, lambda fr, p: price_ema_on(fr, 100)))
    res['No filter'] = dict(dev=score_filter(rows, lambda fr, p: np.ones(len(fr), np.int8)))
    for k, v in res.items():
        print(k, v, flush=True)
    res['_frozen_at'] = pd.Timestamp.utcnow().isoformat()
    json.dump(res, open('FROZEN_stage4.json', 'w'), indent=1, default=float)


def filters_frozen():
    fz = json.load(open('FROZEN_stage4.json'))
    F = {'Model (frozen)': model_fn(tuple(fz['Model']['cfg']))}
    for fam in GRIDS:
        F[f'{fam} {fz[fam]["params"]} (tuned pre-2019)'] = (lambda fr, p, fam=fam, prm=fz[fam]['params']: baseline_on(fr, fam, prm))
    F['EMA200'] = lambda fr, p: price_ema_on(fr, 200)
    F['EMA100'] = lambda fr, p: price_ema_on(fr, 100)
    F['No filter'] = lambda fr, p: np.ones(len(fr), np.int8)
    return F


def test():
    from .stage3 import frame_generic, FX, IDX, STOCKS
    F = filters_frozen()
    cells = []
    o = pd.read_pickle('out_stage2_C.pkl')
    for tf in ('1h', '4h'):
        ser = {}
        for a in COINS:
            x = o[(o.asset == a) & (o.tf == tf)]
            ser[(a, tf)] = (frame3(a, tf), x.p, x.index.min(), x.index.max() + pd.Timedelta(tf))
        cells.append((f'Crypto {tf} (2019-2026)', ser))
    fzc = json.load(open('FROZEN_v3.json'))['C']
    m = lgb.Booster(model_file='trend_model_universal_C.txt')
    for grp, names, tfs in (('FX', FX, ('1h', '4h')), ('Indices', IDX, ('1h', '4h')), ('Stocks', STOCKS, ('1D',))):
        for tf in tfs:
            ser = {}
            for n in names:
                fr = frame_generic(n, tf)
                p = pd.Series(m.predict(fr[fzc['cols']]), fr.index)
                ser[(n, tf)] = (fr, p, fr.index[min(1000, len(fr) - 1)], fr.index[-1])
            cells.append((f'{grp} {tf}', ser))
    out = []
    for cell, ser in cells:
        rows = collect(ser)
        for fname, fn in F.items():
            t, n, mr, wr = score_filter(rows, fn)
            diag = []
            for key, (fr, p, ev, R) in rows.items():
                lo = ser[key][2]
                on = fn(fr, p)
                k0 = fr.index.searchsorted(lo)
                c = fr.close.values[k0:]
                lab2 = oracle_labels(fr.close.values, 2.0)[k0:]
                diag.append(diagnostics(on[k0:], c, lab2))
            dg = pd.DataFrame(diag).mean().to_dict()
            out.append(dict(cell=cell, filter=fname, tstat=t, n_trades=n, mean_R=mr, win_rate=wr, **dg))
        print(cell, 'done', flush=True)
    R = pd.DataFrame(out)
    R.to_pickle('out_stage4.pkl')
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    {'tune': tune, 'test': test}[sys.argv[1]]()

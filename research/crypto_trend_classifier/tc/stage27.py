"""Stage 27: robust SuperTrend settings on every timeframe + volatility sizing (PREREG_stage27_supertrend_all_timeframes.md).
python -m tc.stage27 tune|test"""
import json
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from .labels import oracle_labels
from .s12lib import lag_metrics
from .stage16 import TZ
from . import stage26 as s26

EVAL, EXTRA = s26.EVAL, s26.EXTRA
COINS = EVAL + EXTRA
PRIMARY_TFS = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d']
TFS = ['5m'] + PRIMARY_TFS          # 5m: secondary (amendment 1)
HOURS = {'5m': 1 / 12, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '4h': 4, '8h': 8, '12h': 12, '1d': 24}
PERIODS = [5, 7, 10, 14, 20, 24, 30, 40, 48, 60, 80, 100]
MULTS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
GRID = [(p, m) for p in PERIODS for m in MULTS]
DEV = [(pd.Timestamp(f'{y}-01-01', tz=TZ), pd.Timestamp(f'{y + 1}-01-01', tz=TZ)) for y in range(2019, 2024)]
TEST = (pd.Timestamp('2024-01-01', tz=TZ), pd.Timestamp('2026-09-01', tz=TZ))
COST, TRIM = 0.001, 60


def bars(c, tf):
    src = {'5m': f'bnc/s27/{c}_spot5m.pkl', '15m': f'bnc/s27/{c}_spot15m.pkl', '30m': f'bnc/s27/{c}_spot15m.pkl'}.get(
        tf, f'bnc/s26/{c}_spot1h.pkl')
    d = pd.read_pickle(src)[['open', 'high', 'low', 'close']].astype(float)
    if tf not in ('5m', '15m', '1h'):
        d = d.resample(tf.replace('m', 'min'), origin='epoch', label='left', closed='left').agg(
            dict(open='first', high='max', low='min', close='last')).dropna()
    d.index.name = 'timestamp'
    return d


def mcc(y, s):
    y, s = y.astype(bool), s.astype(bool)
    tp, tn = np.sum(y & s), np.sum(~y & ~s)
    fp, fn = np.sum(~y & s), np.sum(y & ~s)
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def evaluate(args):
    """per setting: per (window, coin) classification metrics, and the equal-weight long-filter pnl per window."""
    tf, windows, coins, sized = args
    bpy = 8760 / HOURS[tf]
    F = {c: bars(c, tf) for c in coins}
    idx = sorted(set().union(*[f.index for f in F.values()]))
    idx = pd.DatetimeIndex(idx)
    rows = []
    pnl = {g: np.zeros(len(idx)) for g in GRID}
    avail = np.zeros(len(idx))
    for c, f in F.items():
        tr = (oracle_labels(f.close.values, 1.0) == 1).astype(np.int64)
        lc = np.log(f.close.values)
        r = np.r_[np.diff(f.close.values) / f.close.values[:-1], 0.0]          # return from t to t+1
        pos_i = idx.get_indexer(f.index)
        avail[pos_i[:-1]] += 1
        if sized:
            n = int(30 * 24 / HOURS[tf])
            sig = pd.Series(np.r_[0.0, r[:-1]]).rolling(n, min_periods=n // 3).std().values
            lev = np.minimum(0.02 * np.sqrt(HOURS[tf] / 24) / sig, 2.0)
            lev = np.nan_to_num(lev)
        wins = [np.flatnonzero((f.index >= lo) & (f.index < hi)) for lo, hi in windows]
        for g in GRID:
            st = s26.st_state(f, g[0], g[1]).astype(np.int64)
            pos = st * lev if sized else st.astype(float)
            x = pos * r - COST * np.abs(np.diff(pos, prepend=0.0))
            x[-1] = 0.0
            pnl[g][pos_i] += x
            for k, w in enumerate(wins):
                if len(w) == 0:
                    continue
                ww = w[:-TRIM] if windows[k] == TEST else w
                if len(ww) < 0.5 * (windows[k][1] - windows[k][0]) / pd.Timedelta(hours=HOURS[tf]) * 0.9:
                    continue
                fpt, mm, md, mg, ns = lag_metrics(st[ww], tr[ww], lc[ww])
                rows.append((tf, g[0], g[1], k, c, mcc(tr[ww], st[ww]), fpt, md, mm))
    R = pd.DataFrame(rows, columns=['tf', 'p', 'm', 'fold', 'coin', 'mcc', 'fpt', 'delay', 'missed'])
    av = np.maximum(avail, 1)
    S = []
    for g in GRID:
        port = pd.Series(pnl[g] / av, index=idx)
        for k, (lo, hi) in enumerate(windows):
            x = port[(port.index >= lo) & (port.index < hi)]
            eq = (1 + x).cumprod()
            S.append((tf, g[0], g[1], k, x.mean() / x.std() * np.sqrt(bpy) if x.std() > 0 else np.nan,
                      float((eq / eq.cummax() - 1).min()), float(eq.iloc[-1] ** (bpy / len(x)) - 1)))
    S = pd.DataFrame(S, columns=['tf', 'p', 'm', 'fold', 'sharpe', 'maxdd', 'cagr'])
    print('done', tf, 'sized' if sized else '', flush=True)
    return R, S


def scores(R, S):
    """per setting: ACC and FILTER, averaged over folds."""
    g = R.groupby(['p', 'm', 'fold'])
    acc = (g.mcc.mean() - 0.1 * np.maximum(0, g.fpt.median() - 2)).groupby(['p', 'm']).mean()
    fil = S.groupby(['p', 'm']).sharpe.mean()
    return acc, fil


def smooth(s):
    M = s.unstack().reindex(index=PERIODS, columns=MULTS).values
    out = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            out[i, j] = np.nanmean(M[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2])
    return pd.DataFrame(out, index=PERIODS, columns=MULTS).stack()


def pick(s):
    s = s.dropna()
    best = s.max()
    cand = s[s >= best - 1e-12].index
    return max(cand, key=lambda x: (x[0], x[1]))


def tune():
    jobs = [(tf, DEV, COINS, False) for tf in TFS]
    with Pool(4) as pool:
        out = pool.map(evaluate, jobs)
    fz = {}
    allR, allS = [], []
    for tf, (R, S) in zip(TFS, out):
        allR.append(R)
        allS.append(S)
        acc, fil = scores(R, S)
        sa, sf = smooth(acc), smooth(fil)
        per_year_acc = [tuple(map(float, (R[R.fold == k].groupby(['p', 'm']).mcc.mean() - 0.1 * np.maximum(
            0, R[R.fold == k].groupby(['p', 'm']).fpt.median() - 2)).idxmax())) for k in range(len(DEV))]
        per_year_fil = [tuple(map(float, S[S.fold == k].set_index(['p', 'm']).sharpe.idxmax())) for k in range(len(DEV))]
        fz[tf] = dict(acc_choice=list(map(float, pick(sa))), filter_choice=list(map(float, pick(sf))),
                      acc_raw_argmax=list(map(float, acc.idxmax())), filter_raw_argmax=list(map(float, fil.idxmax())),
                      acc_dev=float(acc[pick(sa)]), filter_dev=float(fil[pick(sf)]),
                      default_10_3_acc_dev=float(acc[(10, 3.0)]), default_10_3_filter_dev=float(fil[(10, 3.0)]),
                      per_year_acc_argmax=per_year_acc, per_year_filter_argmax=per_year_fil)
        print(tf, json.dumps(fz[tf]), flush=True)
    pd.concat(allR).to_pickle('cache_stage16/s27_dev_R.pkl')
    pd.concat(allS).to_csv('results_stage27_dev_filter.csv', index=False)
    json.dump(dict(choices=fz, frozen_at=str(pd.Timestamp.utcnow())), open('FROZEN_stage27.json', 'w'), indent=1)


def test():
    fz = json.load(open('FROZEN_stage27.json'))['choices']
    jobs = [(tf, [TEST], EVAL, sz) for tf in TFS for sz in (False, True)] + [(tf, [TEST], EXTRA, False) for tf in TFS]
    with Pool(4) as pool:
        out = pool.map(evaluate, jobs)
    res = {(j[0], j[2] == EVAL, j[3]): o for j, o in zip(jobs, out)}
    rows, A_ok, A_def, B_ok, B_def, C_gain = [], 0, 0, 0, 0, []
    for tf in TFS:
        R, S = res[(tf, True, False)]
        acc, fil = scores(R, S)
        _, Sz = res[(tf, True, True)]
        fz_ = fz[tf]
        ac, fc = tuple(fz_['acc_choice']), tuple(fz_['filter_choice'])
        pa = (acc < acc[ac]).mean()
        pf = (fil < fil[fc]).mean()
        szs = Sz.set_index(['p', 'm']).sharpe
        if tf in PRIMARY_TFS:
            A_ok += pa >= 0.75
            A_def += acc[ac] > acc[(10, 3.0)]
            B_ok += pf >= 0.75
            B_def += fil[fc] > fil[(10, 3.0)]
            C_gain.append(szs[fc] - fil[fc])
        m = R.groupby(['p', 'm'])[['mcc', 'fpt', 'delay', 'missed']].median()
        Ssr = S.set_index(['p', 'm'])
        Rx, Sx = res[(tf, False, False)]
        accx, filx = scores(Rx, Sx)
        for name, g in (('ACC choice', ac), ('FILTER choice', fc), ('default (10,3)', (10, 3.0)),
                        ('test-hindsight best ACC', acc.idxmax()), ('test-hindsight best FILTER', fil.idxmax())):
            rows.append(dict(tf=tf, setting=name, p=g[0], m=g[1], acc=acc[g], acc_pct=(acc < acc[g]).mean(),
                             mcc_med=m.loc[g, 'mcc'], fpt_med=m.loc[g, 'fpt'], delay_bars=m.loc[g, 'delay'],
                             delay_hours=m.loc[g, 'delay'] * HOURS[tf], sharpe=fil[g], sharpe_pct=(fil < fil[g]).mean(),
                             maxdd=Ssr.loc[g, 'maxdd'], cagr=Ssr.loc[g, 'cagr'], sharpe_sized=szs[g],
                             maxdd_sized=Sz.set_index(['p', 'm']).loc[g, 'maxdd'],
                             extra_acc=accx[g], extra_sharpe=filx[g]))
    T = pd.DataFrame(rows)
    T.to_csv('results_stage27_test.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.round(3).to_string())
    print(f'A (ACC robust): >=75th pct in {A_ok}/8, beats (10,3) in {A_def}/8 ->', A_ok >= 6 and A_def >= 6)
    print(f'B (FILTER robust): >=75th pct in {B_ok}/8, beats (10,3) in {B_def}/8 ->', B_ok >= 6 and B_def >= 6)
    print(f'C (vol sizing): gains {np.round(C_gain, 3).tolist()}, positive {sum(g > 0 for g in C_gain)}/8, '
          f'median {np.median(C_gain):+.3f} ->', sum(g > 0 for g in C_gain) >= 6 and np.median(C_gain) >= 0.10)
    pd.to_pickle(res, 'cache_stage16/s27_test_res.pkl')


if __name__ == '__main__':
    dict(tune=tune, test=test)[sys.argv[1]]()

"""Stage 6: cross-sectional crash / end-of-bull information (see PREREG_stage6_breadth.md). Run once."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

H, DEV_END, TEST0 = 63, pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-01')
A_COLS = ['dd252', 'r21', 'r63', 'r252', 'sma200', 'lvol21', 'volratio']
X_COLS = ['AR', 'dAR', 'corr63', 'dcorr', 'br200', 'br50', 'dbr200', 'nhnl', 'disp21', 'deep20']


def panel():
    d = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'adj_close'])
    P = d.pivot_table(index='date', columns='ticker', values='adj_close').sort_index()
    P.index = pd.to_datetime(P.index)
    P = P.where(P > 0)
    R = np.log(P).diff().clip(-0.4, 0.4)
    return P, R


def features(P, R):
    idx = np.exp(R.mean(axis=1).fillna(0).cumsum())
    f = pd.DataFrame(index=R.index)
    f['dd252'] = idx / idx.rolling(252).max() - 1
    for n in (21, 63, 252):
        f[f'r{n}'] = np.log(idx / idx.shift(n))
    f['sma200'] = idx / idx.rolling(200).mean() - 1
    rm = np.log(idx).diff()
    v21, v252 = rm.rolling(21).std(), rm.rolling(252).std()
    f['lvol21'] = np.log(v21)
    f['volratio'] = v21 / v252
    # PCA absorption (PC1 share of 252d correlation matrix) and average pairwise correlation 63d
    Rv, ar, c63 = R.values, np.full(len(R), np.nan), np.full(len(R), np.nan)
    for t in range(252, len(R)):
        W = Rv[t - 251:t + 1]
        ok = ~np.isnan(W).any(0)
        if ok.sum() >= 50:
            Z = W[:, ok]
            C = np.corrcoef(Z, rowvar=False)
            C = np.nan_to_num(C)
            ev = np.linalg.eigvalsh(C)
            ar[t] = ev[-1] / ev.sum()
        W = Rv[t - 62:t + 1]
        ok = ~np.isnan(W).any(0) & (np.nanstd(W, 0) > 0)
        if ok.sum() >= 50:
            Z = W[:, ok]
            Z = (Z - Z.mean(0)) / Z.std(0)
            n = Z.shape[1]
            c63[t] = (np.var(Z.sum(1)) - n) / (n * (n - 1))
    f['AR'] = ar
    a = f['AR']
    f['dAR'] = (a.rolling(15).mean() - a.rolling(252).mean()) / a.rolling(252).std()
    f['corr63'] = c63
    f['dcorr'] = f['corr63'] - f['corr63'].rolling(252).mean()
    s200, s50 = P.rolling(200, min_periods=200).mean(), P.rolling(50, min_periods=50).mean()
    f['br200'] = (P > s200)[s200.notna()].sum(1) / s200.notna().sum(1)
    f['br50'] = (P > s50)[s50.notna()].sum(1) / s50.notna().sum(1)
    f['dbr200'] = f['br200'] - f['br200'].shift(21)
    hi, lo = P.rolling(252, min_periods=252).max(), P.rolling(252, min_periods=252).min()
    n = hi.notna().sum(1)
    f['nhnl'] = ((P >= hi) & hi.notna()).sum(1).sub(((P <= lo) & lo.notna()).sum(1)) / n
    f['disp21'] = np.log(P / P.shift(21)).std(axis=1)
    f['deep20'] = ((P / hi - 1) <= -0.2)[hi.notna()].sum(1) / n
    # target: forward 63d max drawdown <= -10%
    iv = idx.values
    fwd = np.full(len(iv), np.nan)
    for t in range(len(iv) - H):
        fwd[t] = iv[t + 1:t + H + 1].min() / iv[t] - 1
    f['fwd_dd'] = fwd
    f['y'] = (fwd <= -0.10).astype(float)
    f.loc[np.isnan(fwd), 'y'] = np.nan
    f['idx'] = idx
    return f


def fit(f, cols):
    dev = f[(f.index >= '1996-01-01') & (f.index < DEV_END)].dropna(subset=cols + ['y'])
    dev = dev.iloc[:-H]  # purge: labels whose window reaches 2010
    sc = StandardScaler().fit(dev[cols])
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(dev[cols]), dev.y)
    ok = f[cols].notna().all(1)
    p = pd.Series(np.nan, f.index)
    p[ok] = m.predict_proba(sc.transform(f.loc[ok, cols]))[:, 1]
    return p, np.quantile(p[dev.index], 0.8), dict(zip(cols, m.coef_[0]))


def block_boot_auc(y, pa, pb, block=126, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    k = int(np.ceil(n / block))
    out = []
    for _ in range(reps):
        st = rng.integers(0, n - block, k)
        ix = (st[:, None] + np.arange(block)).ravel()[:n]
        if y[ix].min() == y[ix].max():
            continue
        out.append(roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pa[ix]))
    return np.quantile(out, [0.05, 0.95])


def exit_test(f, p, thr, te):
    off = (p > thr).astype(int)[te]
    fd = f.fwd_dd[te]
    runs, cur = [], None
    for t, o in off.items():
        if o and cur is None:
            cur = t
        if not o and cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    false = sum(fd.get(r, np.nan) > -0.10 for r in runs)
    crash_days = f.y[te] == 1
    return dict(time_off=off.mean(), crash_days_covered=off[crash_days].mean(), off_runs=len(runs), false_off_runs=int(false))


def dip_events(f, te):
    idx, dd = f.idx, f.dd252
    ev, armed = [], True
    for t in dd.index[dd.notna()]:
        if dd[t] >= -0.001:
            armed = True
        elif armed and dd[t] <= -0.05:
            armed = False
            later = dd[t:]
            rec = later[later >= -0.001]
            end = rec.index[0] if len(rec) else dd.index[-1]
            peak = idx[:t].tail(252).max()
            ev.append(dict(date=t, depth=idx[t:end].min() / peak - 1))
    return pd.DataFrame(ev)


def main():
    P, R = panel()
    f = features(P, R)
    f.to_pickle('out_stage6_features.pkl')
    pa, ta, ca = fit(f, A_COLS)
    pb, tb, cb = fit(f, A_COLS + X_COLS)
    te = (f.index >= TEST0) & f.y.notna() & pa.notna() & pb.notna()
    y = f.y[te].values
    res = dict(n_test=int(te.sum()), base_rate=y.mean(), auc_A=roc_auc_score(y, pa[te]), auc_B=roc_auc_score(y, pb[te]))
    res['diff'] = res['auc_B'] - res['auc_A']
    res['ci90'] = block_boot_auc(y, pa[te].values, pb[te].values).tolist()
    for k, p in (('A', pa), ('B', pb)):
        res[f'brier_skill_{k}'] = 1 - brier_score_loss(y, p[te]) / brier_score_loss(y, np.full(len(y), y.mean()))
    res['exit_A'] = exit_test(f, pa, ta, te)
    res['exit_B'] = exit_test(f, pb, tb, te)
    res['coef_B'] = cb
    res['univariate_auc_test'] = {c: roc_auc_score(y, f[c][te].fillna(f[c][te].median())) for c in X_COLS + A_COLS}
    ev = dip_events(f, te)
    ev['pA'] = pa.reindex(ev.date).values
    ev['pB'] = pb.reindex(ev.date).values
    ev['test'] = ev.date >= TEST0
    pd.to_pickle(dict(res=res, events=ev, pa=pa, pb=pb), 'out_stage6.pkl')
    import json
    print(json.dumps(res, indent=1, default=float))
    print(ev.round(3).to_string())


if __name__ == '__main__':
    main()

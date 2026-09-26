"""Why did ML fail to beat SuperTrend? Diagnostic study on the stage-26 data (exploratory, not a test).
python -m tc.diag26 ceiling|auc|predict|stability|payoff|sizing"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.stats import spearmanr
from . import stage26 as s
from .labels import oracle_labels

EVAL, COINS = s.EVAL, s.COINS
T0, T1, TRIM = s.T0, s.T1, s.TRIM
FZ = json.load(open('FROZEN_stage26.json'))
P = FZ['params']
PRIM = (48, 4)
OUT = 'diag26'
pd.set_option('display.width', 220)


def closes():
    return {c: s.load(c)[['open', 'high', 'low', 'close']] for c in COINS}


def save(df, name):
    df.to_csv(f'results_diag26_{name}.csv')
    print(df.round(4).to_string(), flush=True)


# ---------------------------------------------------------------- 1. where do SuperTrend's errors come from? ceilings
def decompose(state, truth):
    """wrong bars split into LAG (start of a true segment, before the indicator catches up) and WHIPSAW (rest)."""
    n = len(truth)
    ch = np.flatnonzero(np.diff(truth)) + 1
    st, en = np.r_[0, ch], np.r_[ch, n]
    lag = whip = 0
    for a, b in zip(st, en):
        d = truth[a]
        hit = np.flatnonzero(state[a:b] == d)
        if len(hit) == 0:
            lag += b - a
            continue
        h = a + hit[0]
        lag += h - a
        whip += int((state[h:b] != d).sum())
    return lag, whip


def ceiling():
    C = closes()
    EV = s.load_events(*PRIM, EVAL)
    rows, drows = [], []
    for c in EVAL:
        f = C[c]
        tr = s.truth(f)
        w = s.window(f.index, T0, T1, trim=TRIM)
        st = s.st_state(f, *PRIM)
        E = EV[c]
        # perfect meta-labelers: accept an up-flip only if it truly was good (uses the future: an upper bound)
        maj = np.array([np.mean(tr[a:b] == 1) > 0.5 for a, b in zip(E.i.values, np.minimum(E.end.values, len(tr)))])
        good_ret = (E.ret.values - s.RT_COST > 0)
        cands = {'SuperTrend(48,4) 1h': st,
                 'PERFECT meta-label (oracle label)': s.meta_state(st, E.i.values, maj),
                 'PERFECT meta-label (profitable label)': s.meta_state(st, E.i.values, good_ret)}
        for d in (0, 2, 4, 8, 12, 18, 24, 36, 48):
            cands[f'oracle truth delayed {d} bars'] = np.r_[np.full(d, tr[0]), tr[:len(tr) - d]] if d else tr
        for k, v in cands.items():
            m = s.metrics(v[w], tr[w], np.log(f.close.values[w]))
            lag, whip = decompose(v[w], tr[w])
            rows.append(dict(strategy=k, coin=c, mcc=m['mcc'], fpt=m['fpt'], delay=m['delay'],
                             err=float(np.mean(v[w] != tr[w])), lag_share=lag / max(lag + whip, 1)))
    R = pd.DataFrame(rows).groupby('strategy', sort=False).median(numeric_only=True)
    save(R, 'ceiling')


# ---------------------------------------------------------------- 2. can the meta-model rank flips at all? (AUC)
BLOCKS = {'1h price+MTF': ('h1_',), '1h flow': ('h1flow_',), '4h price': ('h4_price_', 'h4_st48'),
          '4h flow': ('h4_flow_',), '4h derivatives': ('h4_deriv_',), '4h cross-market': ('h4_cross_',),
          'event context': ('ev_',)}


def eval_labels(E, tr_full, label):
    if label == 'ret':
        return (E.ret.values - s.RT_COST > 0).astype(int)
    return np.array([np.mean(tr_full[a:b] == 1) > 0.5 for a, b in zip(E.i.values, E.lab_end.values)], dtype=int)


def auc():
    import lightgbm as lgb
    C = closes()
    EV = s.load_events(*PRIM)
    TF = {c: s.truth(C[c]) for c in COINS}
    rows = []
    rng = np.random.default_rng(0)
    for label in ('ret', 'oracle'):
        for Y in range(2022, 2027):
            cut, hi = s.yr(Y), min(s.yr(Y + 1), T1)
            Etr, y = s.train_set(EV, C, cut, label)
            te = pd.concat([E[(E.t >= cut) & (E.t < hi) & E.lab_end_t.notna()] for E in EV.values()])
            yt = np.concatenate([eval_labels(te[te.coin == c], TF[c], label) for c in te.coin.unique()])
            te = pd.concat([te[te.coin == c] for c in te.coin.unique()])
            specs = {'ALL features': s.feat_list(Etr), 'ALL, shuffled training labels (null)': s.feat_list(Etr)}
            for b, pre in BLOCKS.items():
                specs[b] = [x for x in s.feat_list(Etr) if x.startswith(pre)]
            for name, cols in specs.items():
                yy = rng.permutation(y) if 'shuffled' in name else y
                m = lgb.train(s.lgbp(P), lgb.Dataset(Etr[cols].values.astype(np.float32), yy), num_boost_round=P['trees'])
                pr = m.predict(te[cols].values.astype(np.float32))
                rows.append(dict(label=label, year=Y, features=name, auc=roc_auc_score(yt, pr), n_test=len(yt),
                                 base=float(yt.mean())))
            print(label, Y, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('results_diag26_auc_raw.csv', index=False)
    save(R.pivot_table(index='features', columns='label', values='auc', aggfunc='mean').sort_values('ret', ascending=False), 'auc')
    save(R[R.features == 'ALL features'].pivot_table(index='year', columns='label', values='auc'), 'auc_years')


# ---------------------------------------------------------------- 3. what IS predictable: direction vs size vs current state
def predict():
    import lightgbm as lgb
    H, STEP = 24, 4
    parts = []
    for c in COINS:
        f = s.load(c)
        lc = np.log(f.close.values)
        fwd = np.r_[lc[H:] - lc[:-H], np.full(H, np.nan)]
        r1 = np.r_[np.nan, np.diff(lc)]
        sig = pd.Series(r1).rolling(720, min_periods=240).std().values * np.sqrt(H)
        g = f.iloc[::STEP].copy()
        k = np.arange(0, len(f), STEP)
        g['y_dir'] = (fwd[k] > 0).astype(float)
        g.loc[np.isnan(fwd[k]), 'y_dir'] = np.nan
        g['y_size'] = np.abs(fwd[k]) / sig[k]
        g['y_state'] = s.truth(f)[k].astype(float)
        g['coin'] = c
        g['pos'] = k
        parts.append(g.drop(columns=['open', 'high', 'low']))
        print('rows', c, flush=True)
    D = pd.concat(parts)
    feats = [x for x in D.columns if x.startswith(('h1_', 'h1flow_', 'h4_'))]
    prm = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.5,
               bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, seed=0)
    rows = []
    for Y in (2024, 2025, 2026):
        cut, hi = s.yr(Y), min(s.yr(Y + 1), T1)
        tr = D[D.index < cut - pd.Timedelta('3D')]
        te = D[(D.index >= cut) & (D.index < hi) & D.y_dir.notna()]
        te = te[te.coin.isin(EVAL)]
        med = tr.y_size.median()
        for tgt in ('y_dir', 'y_size', 'y_state'):
            if tgt == 'y_state':    # training labels must be final at the cut: oracle on truncated data, minus 60 bars
                ys = []
                for c in COINS:
                    fc = s.load(c)[['close']]
                    past = fc.close.values[fc.index < cut]
                    lab, fin = oracle_labels(past, 1.0, return_final=True)
                    t = tr[tr.coin == c]
                    ok = t.pos.values < fin - 60
                    ys.append(pd.Series(np.where(ok, (lab[np.minimum(t.pos.values, len(lab) - 1)] == 1).astype(float), np.nan), index=t.index))
                ytr = pd.concat(ys).values
            elif tgt == 'y_size':
                ytr = (tr.y_size.values > med).astype(float)
                ytr[np.isnan(tr.y_size.values)] = np.nan
            else:
                ytr = tr.y_dir.values
            ok = ~np.isnan(ytr)
            m = lgb.train(prm, lgb.Dataset(tr.loc[ok, feats].values.astype(np.float32), ytr[ok]), num_boost_round=300)
            pr = m.predict(te[feats].values.astype(np.float32))
            yte = (te.y_size.values > med).astype(int) if tgt == 'y_size' else te[tgt].values.astype(int)
            a = roc_auc_score(yte, pr)
            imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
            rows.append(dict(year=Y, target=tgt, auc=a, top_features=', '.join(imp.index[:5])))
            print(Y, tgt, round(a, 4), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('results_diag26_predictability.csv', index=False)
    print(R.pivot_table(index='target', columns='year', values='auc').round(4).to_string())
    for t in ('y_dir', 'y_size', 'y_state'):
        print(t, '|', R[R.target == t].top_features.iloc[-1])


# ---------------------------------------------------------------- 4. are feature -> label relations stable over time?
def stability():
    C = closes()
    EV = s.load_events(*PRIM)
    TF = {c: s.truth(C[c]) for c in COINS}
    E = pd.concat([E[E.lab_end_t.notna() & (E.t < T1)] for E in EV.values()])
    E['y_ret'] = (E.ret.values - s.RT_COST > 0).astype(int)
    E['y_orc'] = np.concatenate([eval_labels(EV[c][EV[c].lab_end_t.notna() & (EV[c].t < T1)], TF[c], 'oracle') for c in EV])
    feats = s.feat_list(E.drop(columns=['y_ret', 'y_orc']))
    per = {'2019-21': (s.yr(2019), s.yr(2022)), '2022-23': (s.yr(2022), s.yr(2024)), '2024-26': (s.yr(2024), T1)}
    out = {}
    for lab in ('y_ret', 'y_orc'):
        A = {}
        for k, (lo, hi) in per.items():
            e = E[(E.t >= lo) & (E.t < hi)]
            a = {}
            for x in feats:
                v = e[x].values
                ok = np.isfinite(v)
                if ok.sum() > 200 and e[lab].values[ok].std() > 0:
                    a[x] = roc_auc_score(e[lab].values[ok], v[ok]) - 0.5
            A[k] = pd.Series(a)
        A = pd.DataFrame(A).dropna()
        out[lab] = A
        top = A['2019-21'].abs().sort_values(ascending=False).index[:20]
        print(f'\n[{lab}] features: {len(A)}  events per period:',
              {k: int(((E.t >= lo) & (E.t < hi)).sum()) for k, (lo, hi) in per.items()})
        print('median |AUC-0.5| per period:', A.abs().median().round(4).to_dict(), ' max:', A.abs().max().round(3).to_dict())
        print('rank corr of (AUC-0.5) across periods:',
              {f'{a} vs {b}': round(spearmanr(A[a], A[b])[0], 3) for a, b in (('2019-21', '2022-23'), ('2022-23', '2024-26'), ('2019-21', '2024-26'))})
        same = (np.sign(A.loc[top, '2019-21']) == np.sign(A.loc[top, '2024-26'])).mean()
        print(f'top-20 features of 2019-21: same sign in 2024-26 for {same:.0%}')
        print(A.loc[top[:10]].round(3).to_string())
    pd.concat(out, axis=1).to_csv('results_diag26_stability.csv')


# ---------------------------------------------------------------- 5. payoff structure of SuperTrend trades and what META rejected
def payoff():
    C = closes()
    EV = s.load_events(*PRIM, EVAL)
    T = pd.read_pickle(f'{s.OUT}/test_states.pkl')
    rows, allr = [], []
    for c in EVAL:
        E = EV[c]
        f = C[c]
        te = (E.t >= T0) & (E.t < T1)
        cl = f.close.values
        end = np.minimum(E.end.values, len(cl) - 1)
        net = cl[end] / cl[E.i.values] - 1 - s.RT_COST
        allr.append(pd.DataFrame(dict(coin=c, net=net[te.values], acc=T['acc']['META'][c][te.values],
                                      p=T['prob'][c][te.values])))
    A = pd.concat(allr)
    A = A.sort_values('net', ascending=False)
    tot = A.net.sum()
    k10 = max(int(len(A) * 0.1), 1)
    print(f'SuperTrend(48,4) 1h test trades: {len(A)}, win rate {np.mean(A.net > 0):.3f}, mean win {A.net[A.net > 0].mean():.4f}, '
          f'mean loss {A.net[A.net <= 0].mean():.4f}')
    print(f'sum of net returns {tot:.3f}; top 10% of trades contribute {A.net.iloc[:k10].sum():.3f} '
          f'({A.net.iloc[:k10].sum() / tot:.0%} of total); the other 90% sum to {A.net.iloc[k10:].sum():.3f}')
    for nm, g in (('accepted by META', A[A.acc]), ('rejected by META', A[~A.acc])):
        print(f'{nm}: n {len(g)}, win rate {np.mean(g.net > 0):.3f}, mean net {g.net.mean():.4f}, '
              f'share of top-10% winners {g.net.isin(A.net.iloc[:k10]).sum() / k10:.2f}')
    ok = A.p.notna()
    print('Spearman(meta probability, trade net return):', round(spearmanr(A.p[ok], A.net[ok])[0], 4))
    q = pd.qcut(A.p[ok], 5, labels=False)
    print(A[ok].groupby(q).net.agg(['count', 'mean', lambda x: np.mean(x > 0)]).round(4).rename(columns={'<lambda_0>': 'win'}).to_string())
    # cross-coin dependence -> effective number of independent coins
    R = pd.DataFrame({c: C[c].close.pct_change() for c in EVAL}).loc[T0:T1]
    rho = R.corr().values[np.triu_indices(len(EVAL), 1)].mean()
    print(f'mean pairwise hourly return correlation {rho:.2f} -> effective independent coins {len(EVAL) / (1 + (len(EVAL) - 1) * rho):.1f} of 10')
    A.to_csv('results_diag26_trades.csv', index=False)


# ---------------------------------------------------------------- 6. exploratory: use what IS predictable (size) for sizing
def sizing():
    C = closes()
    rows = []
    for name, mode in (('SuperTrend(48,4) 1h, equal size', None), ('SuperTrend(48,4) 1h, inverse-vol size (trailing 30d)', 'vol')):
        pnl = []
        for c in EVAL:
            f = C[c]
            st = s.st_state(f, *PRIM).astype(float)
            r = f.close.pct_change()
            if mode == 'vol':
                sig = r.rolling(720, min_periods=240).std().shift(0)
                w = (0.02 / np.sqrt(24)) / sig          # target ~2 % daily vol per coin
                pos = (st * np.minimum(w.values, 2.0))
            else:
                pos = st
            pos = pd.Series(np.nan_to_num(pos), index=f.index)
            x = pos * r.shift(-1).fillna(0) - s.COST * pos.diff().abs().fillna(0)
            pnl.append(x.loc[T0:T1])
        Pn = pd.concat(pnl, axis=1).fillna(0).mean(axis=1)
        eq = (1 + Pn).cumprod()
        rows.append(dict(strategy=name, sharpe=Pn.mean() / Pn.std() * np.sqrt(8760), vol_ann=Pn.std() * np.sqrt(8760),
                         cagr=eq.iloc[-1] ** (8760 / len(Pn)) - 1, maxdd=(eq / eq.cummax() - 1).min(),
                         calmar=(eq.iloc[-1] ** (8760 / len(Pn)) - 1) / -(eq / eq.cummax() - 1).min()))
    save(pd.DataFrame(rows).set_index('strategy'), 'sizing')


# ---------------------------------------------------------------- 7. how good must a meta-model be? (noisy perfect labeler)
def needed():
    C = closes()
    EV = s.load_events(*PRIM, EVAL)
    rng = np.random.default_rng(0)
    data = []
    for c in EVAL:
        f = C[c]
        tr = s.truth(f)
        E = EV[c]
        maj = np.array([np.mean(tr[a:b] == 1) > 0.5 for a, b in zip(E.i.values, np.minimum(E.end.values, len(tr)))]).astype(float)
        data.append((c, f, tr, s.st_state(f, *PRIM), E.i.values, maj, s.window(f.index, T0, T1, trim=TRIM)))
    base = np.median([matthews_corrcoef(tr[w], st[w]) for c, f, tr, st, u, maj, w in data])
    rows = []
    for sd in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        allm, allp = [], []
        noisy = [(maj + rng.normal(0, sd, len(maj))) for *_, maj, w in data]
        for q in (0.1, 0.2, 0.3, 0.4, 0.5):
            ms = []
            for (c, f, tr, st, u, maj, w), p in zip(data, noisy):
                ms.append(matthews_corrcoef(tr[w], s.meta_state(st, u, p >= np.quantile(p, q))[w]))
            allm.append((np.median(ms), q))
        a = roc_auc_score(np.concatenate([d[5] for d in data]), np.concatenate(noisy))
        best = max(allm)
        rows.append(dict(noise_sd=sd, auc=a, best_mcc=best[0], reject_share=best[1], gain_vs_supertrend=best[0] - base))
    save(pd.DataFrame(rows).set_index('noise_sd'), 'needed_auc')


if __name__ == '__main__':
    dict(ceiling=ceiling, auc=auc, predict=predict, stability=stability, payoff=payoff, sizing=sizing, needed=needed)[sys.argv[1]]()

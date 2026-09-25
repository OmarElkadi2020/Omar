"""Stage 1 development on pre-2019 data only (see PREREG_stage1.md)."""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import matthews_corrcoef
from .v2 import frame2, flip_ratio
from .features import build_bb
from .dataset import load_1h
from .tune import gather, feat_cols, seq_state, ASSETS, TRAIN_TFS

C3 = 'cache3'
FOLDS = [2016, 2017, 2018]
VOLUME = ('vol_z', 'flow_48', 'flow_192')


def frame3(name, tf):
    os.makedirs(C3, exist_ok=True)
    fn = f'{C3}/{name}_{tf}.pkl'
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    f = frame2(name, tf)
    bb = build_bb(load_1h(name), tf).reindex(f.index)
    f = pd.concat([f, bb], axis=1)
    f.to_pickle(fn)
    return f


def cols_for(cand, allcols):
    bbc = [c for c in allcols if c.startswith(('bb_', 'atrp', 'atr_roc')) or c.split('_', 1)[-1].startswith(('bb_', 'atrp', 'atr_roc'))]
    if cand == 'A':
        return [c for c in allcols if c not in bbc]
    if cand == 'B':
        return list(allcols)
    if cand == 'C':
        return [c for c in allcols if not c.startswith(('btc_', 'rel_', 'br_')) and c not in VOLUME]
    raise ValueError(cand)


def make_folds():
    F = []
    for Y in FOLDS:
        tr = gather(f'{Y}-01-01', fr_fn=frame3)
        va = gather(f'{Y + 1}-01-01', assets=ASSETS, since=f'{Y}-01-01', fr_fn=frame3)
        assert tr.index.max() < pd.Timestamp(f'{Y}-01-01', tz='UTC') and va.index.max() < pd.Timestamp('2019-01-01', tz='UTC')
        F.append((tr, va))
        print('fold', Y, 'train', tr.shape, 'valid', va.shape, 'valid assets', sorted(set(va.asset)), flush=True)
    return F


def run(cand, folds, n_trials=60):
    allcols = feat_cols(folds[0][0])
    cols = cols_for(cand, allcols)

    def objective(trial):
        prm = dict(objective='binary', verbosity=-1, num_threads=4, seed=trial.number, bagging_freq=1,
                   learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                   num_leaves=trial.suggest_int('num_leaves', 7, 63, log=True),
                   min_child_samples=trial.suggest_int('min_child_samples', 200, 5000, log=True),
                   feature_fraction=trial.suggest_float('feature_fraction', 0.3, 0.9),
                   bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 0.9),
                   lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 100, log=True),
                   max_depth=trial.suggest_int('max_depth', 3, 8))
        n = trial.suggest_int('n_estimators', 100, 1500, log=True)
        span = trial.suggest_int('smooth_span', 1, 96, log=True)
        h = trial.suggest_float('hysteresis', 0.0, 0.35)
        sc, mc, fl = [], [], []
        for tr, va in folds:
            p = lgb.train(prm, lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), n).predict(va[cols])
            s = seq_state(va, p, span, h)
            yy = va.y.values
            m = np.mean([matthews_corrcoef(yy[va.tf.values == tf], s[va.tf.values == tf]) for tf in TRAIN_TFS])
            f = flip_ratio(va, s)
            sc.append(m - 0.1 * max(0.0, f - 2.0))
            mc.append(m)
            fl.append(f)
        trial.set_user_attr('mcc_folds', mc)
        trial.set_user_attr('flips_folds', fl)
        return float(np.mean(sc))

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    vals = sorted([t.value for t in st.trials], reverse=True)
    res = dict(cand=cand, n_features=len(cols), best=st.best_value, top5_mean=float(np.mean(vals[:5])),
               best_params=st.best_params, best_attrs=st.best_trial.user_attrs, cols=cols)
    print(cand, 'features', len(cols), 'best', round(st.best_value, 4), 'top5', round(res['top5_mean'], 4),
          st.best_trial.user_attrs, flush=True)
    return res


if __name__ == '__main__':
    folds = make_folds()
    out = {c: run(c, folds) for c in sys.argv[1:] or ['A', 'B', 'C']}
    json.dump(out, open('out_dev3.json', 'w'), indent=1, default=float)

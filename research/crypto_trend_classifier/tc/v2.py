"""
Version 2: + market-breadth features, + smoothing tuned under a whipsaw constraint.
Same rules as v1: everything tuned on data < 2019-01-01, frozen, then yearly walk-forward 2019-2026.
Objective (pre-2019 validation year 2018): mean MCC over 1h/4h, minus a penalty when the classifier flips more
than FLIP_CAP times per true trend change (an alert that cries wolf is useless even if its MCC is fine).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import matthews_corrcoef
from .dataset import frame, COINS
from .breadth import with_breadth
from .tune import gather, feat_cols, seq_state, ASSETS, TRAIN_ASSETS, TRAIN_TFS

FLIP_CAP = 2.0
C2 = 'cache2'


def frame2(name, tf):
    os.makedirs(C2, exist_ok=True)
    fn = f'{C2}/{name}_{tf}.pkl'
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    f = with_breadth(frame(name, tf))
    f.to_pickle(fn)
    return f


def flip_ratio(df, s):
    r = []
    for _, idx in df.groupby(['asset', 'tf'], sort=False).indices.items():
        ss, yy = s[idx], df.y.values[idx]
        r.append(np.sum(ss[1:] != ss[:-1]) / max(np.sum(yy[1:] != yy[:-1]), 1))
    return float(np.mean(r))


def tune():
    tr = gather('2018-01-01', fr_fn=frame2)
    va = gather('2019-01-01', assets=ASSETS, since='2018-01-01', fr_fn=frame2)
    cols = feat_cols(tr)
    print('v2 inner train', tr.shape, 'valid', va.shape, 'features', len(cols), flush=True)
    Xtr, ytr, Xva = tr[cols], (tr.y == 1).astype(int), va[cols]

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
        p = lgb.train(prm, lgb.Dataset(Xtr, ytr), n).predict(Xva)
        span = trial.suggest_int('smooth_span', 1, 96, log=True)
        h = trial.suggest_float('hysteresis', 0.0, 0.35)
        s = seq_state(va, p, span, h)
        yy = va.y.values
        mcc = np.mean([matthews_corrcoef(yy[va.tf.values == tf], s[va.tf.values == tf]) for tf in TRAIN_TFS])
        fr_ = flip_ratio(va, s)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('flip_ratio', fr_)
        return float(mcc - 0.1 * max(0.0, fr_ - FLIP_CAP))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80, show_progress_bar=False)
    b = study.best_trial
    print('v2 best', b.value, b.user_attrs, b.params, flush=True)
    old = json.load(open('out_tc_tuning.json'))
    json.dump(dict(model=b.params, valid=b.user_attrs, baselines=old['baselines'],
                   trials=[dict(v=t.value, **t.user_attrs, **t.params) for t in study.trials]),
              open('out_tc_tuning_v2.json', 'w'), indent=1, default=float)


def walkforward(variant='v2'):
    cfg = json.load(open('out_tc_tuning_v2.json'))['model']
    prm = dict(objective='binary', verbosity=-1, num_threads=4, seed=0, bagging_freq=1,
               **{k: v for k, v in cfg.items() if k not in ('n_estimators', 'smooth_span', 'hysteresis')})
    out = []
    for Y in range(2019, 2027):
        cut = f'{Y}-01-01'
        tr = gather(cut, fr_fn=frame2)
        cols = feat_cols(tr)
        m = lgb.train(prm, lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), cfg['n_estimators'])
        lo, hi = pd.Timestamp(cut, tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        assert tr.index.max() < lo
        for a in COINS:
            for tf in ['1h', '2h', '4h']:
                fr = frame2(a, tf)
                te = fr[(fr.index >= lo) & (fr.index < hi)]
                if len(te):
                    out.append(pd.DataFrame(dict(asset=a, tf=tf, p=m.predict(te[cols]), y=te.y.values,
                                                 final=te.final.values, close=te.close.values, year=Y), index=te.index))
        print(variant, Y, len(tr), flush=True)
        if Y == 2026:
            pd.Series(m.feature_importance('gain'), cols).sort_values(ascending=False).to_csv('out_tc_importance_v2.csv')
    pd.concat(out).to_pickle(f'out_tc_oos_{variant}.pkl')


if __name__ == '__main__':
    {'tune': tune, 'wf': walkforward}[sys.argv[1]]()

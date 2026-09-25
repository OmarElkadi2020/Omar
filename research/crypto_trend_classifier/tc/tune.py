"""
Step 1 - everything that is TUNED is tuned on data before 2019-01-01 only, then frozen for all later years:
  * LightGBM hyper-parameters + probability smoothing (Optuna, inner split: train < 2018, validate 2018)
  * every baseline's parameters (grid search, same data)
Nothing from 2019+ is ever seen here.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import matthews_corrcoef
from .dataset import frame, train_rows, COINS
from .evaluate import smooth_state, BASELINE_GRID, BASELINE_FN

ASSETS = ['BTC_BITSTAMP'] + COINS
TRAIN_ASSETS = ASSETS + [a + '_INV' for a in ASSETS]  # upside-down copies, training only
TRAIN_TFS = ['1h', '4h']
NON_FEAT = {'open', 'high', 'low', 'close', 'y', 'final', 'asset', 'tf'}


def feat_cols(fr):
    return [c for c in fr.columns if c not in NON_FEAT]


def gather(cutoff, tfs=TRAIN_TFS, assets=TRAIN_ASSETS, since=None, fr_fn=frame):
    parts = []
    for a in assets:
        for tf in tfs:
            tr = train_rows(fr_fn(a, tf), pd.Timestamp(cutoff, tz='UTC'))
            if since is not None:
                tr = tr[tr.index >= pd.Timestamp(since, tz='UTC')]
            if len(tr):
                parts.append(tr)
    return pd.concat(parts) if parts else None


def seq_state(df, p, span, h):
    """smooth per (asset, tf) sequence, never across series."""
    s = np.empty(len(df), np.int8)
    for _, idx in df.groupby(['asset', 'tf'], sort=False).indices.items():
        s[idx] = smooth_state(p[idx], span, h)
    return s


def main():
    tr = gather('2018-01-01')
    va = gather('2019-01-01', assets=ASSETS, since='2018-01-01')
    cols = feat_cols(tr)
    print('inner train', tr.shape, 'valid', va.shape, 'features', len(cols), flush=True)
    Xtr, ytr = tr[cols], (tr.y == 1).astype(int)
    Xva, yva = va[cols], (va.y == 1).astype(int)

    def objective(trial):
        prm = dict(objective='binary', verbosity=-1, num_threads=4, seed=trial.number,
                   learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                   num_leaves=trial.suggest_int('num_leaves', 7, 63, log=True),
                   min_child_samples=trial.suggest_int('min_child_samples', 200, 5000, log=True),
                   feature_fraction=trial.suggest_float('feature_fraction', 0.3, 0.9),
                   bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 0.9), bagging_freq=1,
                   lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 100, log=True),
                   max_depth=trial.suggest_int('max_depth', 3, 8))
        n = trial.suggest_int('n_estimators', 100, 1500, log=True)
        m = lgb.train(prm, lgb.Dataset(Xtr, ytr), n)
        p = m.predict(Xva)
        span = trial.suggest_int('smooth_span', 1, 48, log=True)
        h = trial.suggest_float('hysteresis', 0.0, 0.25)
        s = seq_state(va, p, span, h)
        yy = va.y.values
        # mean MCC over the two timeframes so neither dominates
        sc = [matthews_corrcoef(yy[va.tf.values == tf], s[va.tf.values == tf]) for tf in TRAIN_TFS]
        return float(np.mean(sc))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    print('best MCC (validation 2018):', study.best_value, study.best_params, flush=True)

    # baselines: pick the best parameters on the same pre-2019 data (train + valid together), per timeframe
    allpre = pd.concat([tr, va])
    base_best = {}
    for tf in TRAIN_TFS + ['2h']:
        base_best[tf] = {}
        frames = [(a, frame(a, tf)) for a in ASSETS]
        rows_of = {a: train_rows(fr, pd.Timestamp('2019-01-01', tz='UTC')) for a, fr in frames}
        for name, grid in BASELINE_GRID.items():
            best, bp = -1, None
            for prm in grid:
                ys, ss = [], []
                for a, fr in frames:
                    rows = rows_of[a]
                    if not len(rows):
                        continue
                    s = pd.Series(BASELINE_FN[name](fr, prm), fr.index).reindex(rows.index).values
                    ys.append(rows.y.values)
                    ss.append(s)
                sc = matthews_corrcoef(np.concatenate(ys), np.concatenate(ss))
                if sc > best:
                    best, bp = sc, prm
            base_best[tf][name] = dict(params=bp, pre2019_mcc=best)
            print(tf, name, bp, round(best, 3), flush=True)
    json.dump(dict(model=study.best_params, model_valid_mcc=study.best_value, baselines=base_best,
                   trials=[dict(v=t.value, **t.params) for t in study.trials]),
              open('out_tc_tuning.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()

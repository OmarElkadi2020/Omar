"""Step 3 - out-of-sample scoring 2019-2026 of the model variants vs the (pre-2019-tuned) baselines."""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from .dataset import frame, COINS
from .evaluate import metrics, smooth_state, BASELINE_FN
sys.path.insert(0, '.')
from trend_eval import regime_classifier  # noqa: E402  (the XAUSTBreakFree 4h regime, 24x7 session for crypto)

BPY = {'1h': 8760, '2h': 4380, '4h': 2190}


def states_for_series(fr_oos, variant_p, cfg, base_cfg, tf, fr_full):
    S = {}
    S['Model (LightGBM, walk-forward)'] = smooth_state(variant_p, cfg['smooth_span'], cfg['hysteresis'])
    for name, bc in base_cfg[tf].items():
        full = pd.Series(BASELINE_FN[name](fr_full, bc['params']), fr_full.index)
        S[f'{name} {bc["params"]}'] = full.reindex(fr_oos.index).values.astype(np.int8)
    g = fr_full[['open', 'high', 'low', 'close']]
    reg, _ = regime_classifier(g, '4h', '24x7')
    reg = pd.Series(reg, g.index).replace(0, np.nan).ffill().fillna(1).astype(np.int8)
    S['Strategy regime (4h SuperTrend+EMA)'] = reg.reindex(fr_oos.index).values
    S['Always long'] = np.ones(len(fr_oos), np.int8)
    return S


def main(variants=('full', 'btc_only', 'no_ctx')):
    tun = json.load(open('out_tc_tuning.json'))
    cfg, base_cfg = tun['model'], tun['baselines']
    rows, yearly = [], []
    oos = {v: pd.read_pickle(f'out_tc_oos_{v}.pkl') for v in variants}
    for tf in ['1h', '2h', '4h']:
        for a in COINS:
            fr_full = frame(a, tf)
            base = oos['full']
            o = base[(base.asset == a) & (base.tf == tf)]
            if not len(o):
                continue
            keep = o.final.values
            S = states_for_series(o, o.p.values, cfg, base_cfg, tf, fr_full)
            for v in variants[1:]:
                ov = oos[v]
                ov = ov[(ov.asset == a) & (ov.tf == tf)].reindex(o.index)
                S[f'Model variant: {v}'] = smooth_state(ov.p.values, cfg['smooth_span'], cfg['hysteresis'])
            y, c = o.y.values[keep], o.close.values[keep]
            for name, s in S.items():
                m = metrics(s[keep], y, c, bars_per_year=BPY[tf])
                m.update(asset=a, tf=tf, clf=name)
                rows.append(m)
                for Y in sorted(set(o.year)):
                    k2 = keep & (o.year.values == Y)
                    if k2.sum() > 500:
                        yearly.append(dict(asset=a, tf=tf, clf=name, year=Y,
                                           mcc=matthews_corrcoef(o.y.values[k2], s[k2]), acc=np.mean(o.y.values[k2] == s[k2])))
        print(tf, 'done', flush=True)
    R = pd.DataFrame(rows)
    Yr = pd.DataFrame(yearly)
    R.to_pickle('out_tc_report_rows.pkl')
    Yr.to_pickle('out_tc_report_yearly.pkl')


if __name__ == '__main__':
    main()

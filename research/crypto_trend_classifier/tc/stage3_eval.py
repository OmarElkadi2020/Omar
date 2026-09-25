"""Scores stage 3 exactly against the pass criteria in PREREG_stage3.md."""
import pandas as pd
import numpy as np


def main():
    R = pd.read_pickle('out_stage3_rows.pkl')
    F = pd.read_pickle('out_stage3_frontier.pkl')
    out = []
    for (g, tf), r in R.groupby(['group', 'tf']):
        mean = r.groupby('clf')[['mcc', 'flips_per_oracle_flip', 'flip_precision', 'missed_move_frac']].mean()
        base = mean.drop(index='Model')
        p1 = mean.loc['Model', 'mcc'] > base.mcc.max()
        f = F[(F.group == g) & (F.tf == tf)]
        f = f.groupby(['method', 'params'])[['mcc', 'flips']].mean().reset_index()
        front = {m: d[d.flips <= 2].mcc.max() for m, d in f.groupby('method')}
        fb = max(v for k, v in front.items() if k != 'Model' and v == v)
        p2 = front.get('Model', -1) > fb
        piv = r.pivot_table(index='asset', columns='clf', values='mcc')
        bestb = piv.drop(columns='Model').max(axis=1)
        share = float((piv['Model'] > bestb).mean())
        p3 = share >= 0.6
        # secondary: hindsight-best parameter per baseline family on this asset class (upper bound)
        hind = f[f.method != 'Model'].groupby('method').mcc.max()
        out.append(dict(group=g, tf=tf, n=len(piv), model_mcc=mean.loc['Model', 'mcc'], model_flips=mean.loc['Model', 'flips_per_oracle_flip'],
                        best_frozen_baseline=base.mcc.idxmax(), best_frozen_baseline_mcc=base.mcc.max(),
                        P1=p1, model_front_le2=front.get('Model'), best_baseline_front_le2=fb, P2=p2,
                        share_instruments_model_wins=share, P3=p3,
                        hindsight_best_baseline_mcc=hind.max(), hindsight_best_family=hind.idxmax()))
        print(f'\n=== {g} {tf}  ({len(piv)} instruments)')
        print(mean.round(3).to_string())
    S = pd.DataFrame(out)
    S.to_csv('out_stage3_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n', S.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

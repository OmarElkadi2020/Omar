import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .dataset import frame, COINS
from .evaluate import smooth_state, BASELINE_FN

BLUE, ORANGE, AQUA, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948'
INK, INK2, GRID, SURF, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#f0efec'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 2})


def pareto(df):
    d = df.sort_values('flips')
    best, xs, ys = -1, [], []
    for f, m in zip(d.flips, d.mcc):
        if m > best:
            best = m
            xs.append(f)
            ys.append(m)
    return xs, ys


def main():
    t1 = json.load(open('out_tc_tuning.json'))
    t2 = json.load(open('out_tc_tuning_v2.json'))
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])

    # A: example
    ax = fig.add_subplot(gs[0, :])
    o2 = pd.read_pickle('out_tc_oos_v2.pkl')
    x = o2[(o2.asset == 'BTCUSDT') & (o2.tf == '1h')]
    fr = frame('BTCUSDT', '1h')
    s_m = pd.Series(smooth_state(x.p.values, t2['model']['smooth_span'], t2['model']['hysteresis']), x.index)
    s_st = pd.Series(BASELINE_FN['SuperTrend'](fr, t1['baselines']['1h']['SuperTrend']['params']), fr.index)
    w = x.loc['2024-02-01':'2024-09-30']
    c = w.close.values
    xi = np.arange(len(w))
    ax.plot(xi, c, color=INK, lw=1)
    lo, hi = c.min(), c.max()
    band = (hi - lo) * 0.06
    for j, (lab, s) in enumerate([('Ideal (look-ahead)', w.y.values), ('Model v2 (real time)', s_m.reindex(w.index).values),
                                  ('SuperTrend 48/4', s_st.reindex(w.index).values)]):
        y0 = lo - (j + 1) * band * 1.25
        ax.fill_between(xi, y0, y0 + band, where=s == 1, color=BLUE, step='post', lw=0)
        ax.fill_between(xi, y0, y0 + band, where=s == -1, color=RED, step='post', lw=0)
        ax.text(-8, y0 + band / 2, lab, ha='right', va='center', color=INK2, fontsize=9)
    ax.set_xlim(-2, len(w))
    mstart = pd.Series(np.arange(len(w)), w.index).groupby(w.index.to_period('M')).first()
    ax.set_xticks(mstart.values)
    ax.set_xticklabels([p.strftime('%b %Y') for p in mstart.index])
    ax.set_yticks([])
    ax.set_title('BTC 1h, Feb–Sep 2024 (out-of-sample): blue = up-trend, red = down-trend', loc='left')

    # B: frontier
    ax = fig.add_subplot(gs[1, 0])
    F = pd.read_pickle('out_tc_frontier_v2_1h.pkl')
    cols = {'Model': BLUE, 'SuperTrend': ORANGE, 'Online directional-change': AQUA, 'EMA cross': YELLOW}
    for m, col in cols.items():
        xs, ys = pareto(F[F.method == m])
        ax.plot(xs, ys, color=col, marker='o', ms=4, label=m if m != 'Model' else 'Model v2 (all smoothing levels)')
    R = pd.read_pickle('out_tc_final_rows.pkl')
    r = R[R.tf == '1h'].groupby('clf')[['mcc', 'flips_per_oracle_flip']].mean()
    mv = r.loc['Model v2 (breadth + whipsaw cap)']
    ax.scatter([mv.flips_per_oracle_flip], [mv.mcc], s=160, facecolor='none', edgecolor=INK, lw=1.5, zorder=5)
    ax.annotate('frozen pre-2019 setting', (mv.flips_per_oracle_flip, mv.mcc), xytext=(40, -60), arrowprops=dict(arrowstyle='-', color=INK2, lw=.8), textcoords='offset points', color=INK2, fontsize=9)
    for L, v in [(24, 0.70), (48, 0.466)]:
        ax.axhline(v, color=INK2, ls=':', lw=1)
        ax.text(8.5, v + 0.01, f'ideal labeler, {L}h late', color=INK2, fontsize=8, ha='right')
    ax.set_xscale('log')
    ax.set_xticks([1, 1.5, 2, 3, 5, 8])
    ax.set_xticklabels(['1', '1.5', '2', '3', '5', '8'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.9, 9)
    ax.set_ylim(0.1, 0.75)
    ax.set_xlabel('flips per true trend change (lower = fewer false alarms)')
    ax.set_ylabel('MCC vs ideal labels (1 = perfect)')
    ax.set_title('1h, 10 coins, 2019–2026 out-of-sample', loc='left')
    ax.legend(fontsize=8, loc='upper left', frameon=False)

    # C: yearly
    ax = fig.add_subplot(gs[1, 1])
    D = pd.read_pickle('out_tc_yearly_v2.pkl')
    d = D[D.tf == '1h'].groupby('year')[['model', 'SuperTrend', 'Online directional-change']].mean()
    for col, cc, lab in [('model', BLUE, 'Model v2'), ('SuperTrend', ORANGE, 'SuperTrend 48/4'), ('Online directional-change', AQUA, 'Online directional-change')]:
        ax.plot(d.index, d[col], color=cc, marker='o', ms=5, label=lab)
    ax.set_ylim(0.2, 0.5)
    ax.set_ylabel('MCC (mean of 10 coins)')
    ax.set_title('1h: every year retrained only on the past', loc='left')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    # D: per coin, 1h
    ax = fig.add_subplot(gs[2, :])
    r = R[R.tf == '1h'].pivot_table(index='asset', columns='clf', values='mcc')
    base_cols = [c for c in r.columns if c.startswith(('SuperTrend', 'Online', 'EMA cross'))]
    r['best classic'] = r[base_cols].max(axis=1)
    r = r.sort_values('Model v2 (breadth + whipsaw cap)', ascending=False)
    xi = np.arange(len(r))
    for j, (col, cc, lab) in enumerate([('Model v2 (breadth + whipsaw cap)', BLUE, 'Model v2 (trained on all coins)'),
                                         ('Model v2 trained on BTC only', AQUA, 'Model v2 trained on BTC only (alts unseen)'),
                                         ('best classic', ORANGE, 'Best classic indicator for that coin (picked after the fact)')]):
        ax.bar(xi + (j - 1) * 0.27, r[col].values, 0.25, color=cc, label=lab)
    ax.set_xticks(xi)
    ax.set_xticklabels([a.replace('USDT', '') for a in r.index])
    ax.set_ylabel('MCC, 1h, 2019–2026')
    ax.set_ylim(0.25, 0.5)
    ax.set_title('Per coin', loc='left')
    ax.legend(fontsize=8, frameon=False, ncol=3, loc='upper right')
    plt.tight_layout()
    plt.savefig('/home/user/Omar/research/crypto_trend_classifier/results.png', dpi=100)


if __name__ == '__main__':
    main()

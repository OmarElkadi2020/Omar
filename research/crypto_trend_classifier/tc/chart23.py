"""Charts for stage 23 (India, survivorship-free): equity, drawdown, alpha per year, robustness."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .stage16 import CD

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'


def style(ax, title):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', color=INK, fontsize=11.5, fontweight='bold', pad=10)


def main():
    R = pd.read_pickle(f'{CD}/india_sf17_returns.pkl').rename(columns={'vm_long_short': 'long_only'})
    R = R[['long_only', 'long_short', 'MKT']].dropna()
    eq = np.exp(np.log1p(R).cumsum())
    yrs = pd.read_csv('results_stage23_india_sf_years.csv')
    ex = pd.read_csv('results_stage23_india_sf_extra.csv')
    main_ = pd.read_csv('results_stage23_india_sf.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), facecolor=SURF)
    # 1 equity
    ax = axes[0, 0]
    style(ax, 'Growth of ₹1 (log scale), 2014–2026, after costs')
    ends = []
    for c, col, lab in (('long_only', S1, 'Model long-only (top 20%)'), ('MKT', S2, 'Equal-weight market'),
                        ('long_short', S3, 'Model long-short')):
        ax.plot(eq.index, eq[c], color=col, lw=2, label=lab)
        ends.append((eq[c].iloc[-1], col, lab.split(' (')[0]))
    ends.sort()
    prev = None
    for v, col, lab in ends:                     # nudge end labels apart when values are close
        dy = 0 if prev is None or v / prev > 1.12 else 11
        ax.annotate(f'₹{v:.1f}', (eq.index[-1], v), xytext=(6, dy), textcoords='offset points',
                    color=INK, fontsize=9.5, va='center', fontweight='bold')
        prev = v
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'₹{v:g}'))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper left')
    ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta('420D'))
    # 2 drawdown
    ax = axes[0, 1]
    style(ax, 'Drawdown from peak')
    for c, col, lab in (('long_only', S1, 'Model long-only'), ('MKT', S2, 'Equal-weight market')):
        dd = eq[c] / eq[c].cummax() - 1
        ax.plot(dd.index, dd * 100, color=col, lw=2, label=f'{lab} (worst {dd.min() * 100:.0f}%)')
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='lower left')
    # 3 alpha per year
    ax = axes[1, 0]
    style(ax, 'Long-only alpha per year (after known factors)')
    a = yrs.set_index('year').alpha_ann * 100
    bars = ax.bar(a.index.astype(str), a.values, color=S1, width=0.7)
    for b in bars:
        b.set_edgecolor(SURF)
        b.set_linewidth(2)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(main_.alpha_ann.iloc[0] * 100, color=INK2, lw=1, ls='--')
    ax.text(len(a) - 0.5, main_.alpha_ann.iloc[0] * 100 + 0.6, f'average {main_.alpha_ann.iloc[0] * 100:.1f}%/yr',
            color=INK2, fontsize=9, ha='right')
    for x, v in zip(bars, a.values):
        if abs(v) >= 9:
            ax.text(x.get_x() + x.get_width() / 2, v + 0.5, f'{v:.0f}%', ha='center', color=INK, fontsize=9)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.tick_params(axis='x', labelrotation=0, labelsize=8.5)
    # 4 robustness (t-stat)
    ax = axes[1, 1]
    style(ax, 'Is the alpha statistically real? (t-stat, 3.0 = pre-registered bar)')
    lab = {'long-only (primary, delist -30%)': 'Primary (delisted −30%)',
           'long-only, delisting return 0%': 'Delisted 0%', 'long-only, delisting return -100%': 'Delisted −100%',
           'long-only, survivors only (bias check)': 'Survivors only',
           'long-only, top-300 liquid': 'Top-300 liquid', 'long-only, top-200 liquid': 'Top-200 liquid',
           'long-only, top-100 liquid': 'Top-100 liquid'}
    ex = ex[ex.test.isin(lab)].copy()
    ex['name'] = ex.test.map(lab)
    extra = pd.DataFrame([dict(name='Costs 0.25%', t=main_.set_index('book').alpha_t['long-only, costs 25bp']),
                          dict(name='+1 day delay', t=main_.set_index('book').alpha_t['long-only +1 day delay'])])
    T = pd.concat([ex[['name', 't']], extra]).iloc[::-1]
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    bars = ax.barh(T.name, T.t, color=[S1 if v >= 3 else '#9fc2ec' for v in T.t], height=0.62)
    for b in bars:
        b.set_edgecolor(SURF)
        b.set_linewidth(2)
    ax.axvline(3.0, color=INK, lw=1.2, ls='--')
    ax.axvline(0, color=INK2, lw=1)
    for b, v in zip(bars, T.t):
        ax.text(v + (0.08 if v >= 0 else -0.08), b.get_y() + b.get_height() / 2, f'{v:.2f}', va='center',
                ha='left' if v >= 0 else 'right', color=INK, fontsize=9)
    ax.tick_params(axis='y', labelsize=9.5, colors=INK)
    ax.set_xlim(min(-1.5, T.t.min() - 0.6), max(4.2, T.t.max() + 0.6))
    fig.suptitle('India NSE — model long-only picks, survivorship-free data (delisted stocks included)',
                 x=0.012, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.text(0.012, 0.935, 'Out-of-sample walk-forward, frozen model, next-open execution, 0.15% costs. Alpha = return left after '
             'market, size, momentum, reversal, low-vol and trend factors.', color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.5, w_pad=2.5)
    fig.savefig('stage23_india_sf.png', dpi=130, facecolor=SURF)


if __name__ == '__main__':
    main()

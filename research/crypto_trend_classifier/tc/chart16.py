import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .stage16 import CD


def main(markets):
    fig, axes = plt.subplots(1, len(markets), figsize=(7 * len(markets), 4.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, markets):
        R = pd.read_pickle(f'{CD}/{m}_returns.pkl')
        if m == 'india17':
            R = R.rename(columns={'vm_long_short': 'long_only_primary'})
        for c, lab in (('long_short', 'model long-short (net)'), ('long_only', 'model long-only top 20% (net)'),
                       ('long_only_primary', 'model long-only top 20% (net, primary)'),
                       ('MKT', 'equal-weight universe'), ('BTC' if 'BTC' in R else 'UMD', 'BTC' if 'BTC' in R else 'UMD momentum')):
            if c not in R:
                continue
            ax.plot(np.log1p(R[c].clip(lower=-0.99)).cumsum().apply(np.exp), label=lab)
        ax.set_yscale('log'); ax.set_title({'crypto': 'Crypto (Binance, incl. delisted) 2022-26', 'india17': 'India NSE top-500 2011-26'}.get(m, m) + ': out-of-sample'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig('stage16_' + '_'.join(markets) + '.png', dpi=110)


if __name__ == '__main__':
    main(sys.argv[1:])

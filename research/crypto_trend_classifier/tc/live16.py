"""Stage-16 ranking rebuilt on data/universe_4h (the original bnc/k4h cache is not in the repo).
python -m tc.live16 check   # rebuild + frozen-procedure test 2022-01 … 2026-09, to compare with the published result
python -m tc.live16 rank    # today's ranking with the frozen procedure (model fitted on data before 1 Jan this year)"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from . import stage16 as s16

UNIV = 'data/universe_4h'


def load_crypto_parquet():
    O, H, L, C, V, C4, H4, L4 = {}, {}, {}, {}, {}, {}, {}, {}
    for f in sorted(glob.glob(f'{UNIV}/*.parquet')):
        s = os.path.basename(f)[:-8]
        if s16.EXCL.search(s) or s in s16.EXCL_NAMES:
            continue
        d = pd.read_parquet(f).astype(float)
        d = d[(d.close > 0) & (d.volume > 0)]
        if len(d) < 6 * 90:
            continue
        g = d.resample('1D').agg(dict(open='first', high='max', low='min', close='last', volume='sum')).dropna()
        O[s], H[s], L[s], C[s], V[s] = g.open, g.high, g.low, g.close, g.volume * g.close
        C4[s], H4[s], L4[s] = d.close, d.high, d.low
    P = {k: pd.DataFrame(v).sort_index() for k, v in dict(O=O, H=H, L=L, C=C, DV=V).items()}
    P4 = {k: pd.DataFrame(v).sort_index() for k, v in dict(C=C4, H=H4, L=L4).items()}
    ex = P4['C'][P4['C'].index.hour == 0]
    ex.index = ex.index.normalize() - pd.Timedelta('1D')
    P['EX'] = ex.reindex(P['C'].index)
    return P, P4


def frozen():
    return json.load(open('prereg/FROZEN_stage16_crypto.json'))['params']


def ensure_build():
    if not os.path.exists(f'{s16.CD}/crypto_X.pkl'):
        s16.load_crypto = load_crypto_parquet
        s16.build('crypto')


def check():
    ensure_build()
    p = frozen()
    X, D = s16.load('crypto')
    parts = []
    for Y in range(2022, 2027):
        lo, hi = pd.Timestamp(f'{Y}-01-01', tz='UTC'), pd.Timestamp(f'{Y + 1}-01-01', tz='UTC')
        m, cols = s16.fit(X, p['H'], lo, p, 1)
        parts.append(s16.predict(m, cols, X, lo, hi))
        print('fitted', Y, flush=True)
    S = pd.concat(parts)
    S.to_pickle(f'{s16.CD}/crypto_scores_rebuilt.pkl')
    out = []
    for name, (t0, t1) in (('published test window 2022-01 … 2026-08', ('2022-01-01', '2026-09-01')),
                           ('after the published test 2026-09 … now', ('2026-09-01', '2027-01-01'))):
        t0, t1 = pd.Timestamp(t0, tz='UTC'), pd.Timestamp(t1, tz='UTC')
        ls, lo_, F, to, tol = s16.evaluate_scores(S[(S.index.get_level_values(0) >= t0) & (S.index.get_level_values(0) < t1)],
                                                  D, 'crypto', p['hold'], t0, t1)
        for book, x in (('long-short', ls), ('long-only', lo_)):
            a, t, _ = s16.alpha(x, F) if len(x) > 60 else (np.nan, np.nan, None)
            out.append(dict(window=name, book=book, days=len(x), alpha_ann=a * 365, alpha_t=t, **s16.stats(x, 365)))
    R = pd.DataFrame(out)
    R.to_csv('results_live16_check.csv', index=False)
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())


def rank():
    ensure_build()
    p = frozen()
    X, D = s16.load('crypto')
    last = X.index.get_level_values(0).max()
    cut = pd.Timestamp(f'{last.year}-01-01', tz='UTC')
    m, cols = s16.fit(X, p['H'], cut, p, 1)
    S = s16.predict(m, cols, X, last - pd.Timedelta(days=p['hold'] + 2), last + pd.Timedelta(days=1)).unstack()
    w = s16.book(S, long_only=True).rolling(p['hold'], min_periods=1).mean()   # the frozen book: top 20 %, held 3 days
    today = S.iloc[-1].dropna().sort_values(ascending=False)
    R = pd.DataFrame({'score': today, 'rank': np.arange(1, len(today) + 1),
                      'pct': (today.rank(pct=True) * 100).round(0), 'book_weight': w.iloc[-1].reindex(today.index)})
    R.index.name = 'asset'
    R.attrs['date'] = str(last.date())
    R.to_csv('results_live16_rank.csv')
    print('signal date', last.date(), '| model fitted on data before', cut.date(), '| eligible', len(today))
    print(R.head(25).round(4).to_string())
    print('...')
    print(R.tail(10).round(4).to_string())


if __name__ == '__main__':
    {'check': check, 'rank': rank}[sys.argv[1]]()

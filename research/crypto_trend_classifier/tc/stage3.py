"""Stage 3: frozen crypto-trained universal model on stocks & FX (see PREREG_stage3.md). Run once."""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from .features import build, build_bb
from .labels import oracle_labels
from .dev3 import frame3
from .tune import gather
from .stage2 import params
from .evaluate import smooth_state, BASELINE_FN, BASELINE_GRID, metrics

FX = ['EUR_USD', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'EUR_JPY', 'AUD_JPY']
IDX = ['SPX500_USD', 'NAS100_USD', 'US2000_USD', 'UK100_GBP', 'JP225_USD', 'AU200_AUD', 'FR40_EUR', 'NL25_EUR', 'DAX']
STOCKS = ('AAPL MSFT NVDA GOOGL AMZN META BRK-B AVGO TSLA LLY JPM V WMT XOM UNH MA ORCL COST HD PG JNJ NFLX BAC ABBV CRM '
          'KO CVX MRK AMD PEP TMO ADBE LIN CSCO ACN MCD WFC ABT IBM GE DIS QCOM TXN INTU AMGN CAT PM VZ NOW GS').split()
C4 = 'cache_stage3'


def load(name):
    if name == 'DAX':
        d = pd.read_csv('data/GRXEUR_histdata_h1.csv', index_col=0, parse_dates=True)
        d.index = d.index.tz_localize('Etc/GMT+5').tz_convert('UTC')  # histdata = EST without DST
    elif name in FX + IDX:
        d = pd.read_csv(f'data/{name}_oanda_h1.csv', index_col=0, parse_dates=True)
        d.index = d.index.tz_localize('UTC')
    else:
        d = STK[STK.ticker == name].set_index('date')[['open', 'high', 'low', 'close', 'volume']].sort_index()
        d.index = pd.to_datetime(d.index).tz_localize('UTC')
        d = d[(d[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    return d[['open', 'high', 'low', 'close', 'volume']].astype(float)


def frame_generic(name, tf):
    os.makedirs(C4, exist_ok=True)
    fn = f'{C4}/{name}_{tf}.pkl'
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    d = load(name)
    base, X = build(d, tf)
    X = pd.concat([X, build_bb(d, tf).reindex(X.index)], axis=1)
    lab, fin = oracle_labels(base.close.values, 1.0, return_final=True)
    f = pd.concat([base[['open', 'high', 'low', 'close']], X], axis=1)
    f['y'] = lab
    f['final'] = np.arange(len(f)) <= fin
    f['asset'] = name
    f['tf'] = tf
    f.to_pickle(fn)
    return f


def main():
    global STK
    STK = pd.read_parquet('data/sp_prices.parquet', columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    fz = json.load(open('FROZEN_v3.json'))['C']
    cols, p = fz['cols'], fz['params']
    tun = json.load(open('out_tc_tuning.json'))['baselines']
    tr = gather('2026-09-24', fr_fn=frame3)  # crypto only
    m = lgb.train(params(p), lgb.Dataset(tr[cols], (tr.y == 1).astype(int)), p['n_estimators'])
    m.save_model('trend_model_universal_C.txt')
    rows, front = [], []
    groups = [('FX', FX, ['1h', '4h']), ('Indices', IDX, ['1h', '4h']), ('Stocks', [s for s in STOCKS if s in set(STK.ticker)], ['1D'])]
    for grp, names, tfs in groups:
        for tf in tfs:
            btf = '4h' if tf == '1D' else tf
            for n in names:
                f = frame_generic(n, tf)
                ev = f.iloc[1000:]
                ev = ev[ev.final]
                if len(ev) < 1000:
                    continue
                prob = m.predict(f[cols])
                S = {'Model': smooth_state(prob, p['smooth_span'], p['hysteresis'])}
                for bn in BASELINE_FN:
                    S[bn] = BASELINE_FN[bn](f, tun[btf][bn]['params'])
                pos = f.index.get_indexer(ev.index)
                for k, s in S.items():
                    mt = metrics(np.asarray(s)[pos], ev.y.values, ev.close.values)
                    mt.update(group=grp, tf=tf, asset=n, clf=k)
                    rows.append(mt)
                # frontier material: model smoothing sweep + every baseline parameter
                for span in (1, 2, 4, 8, 16, 32, 64):
                    for h in (0, .1, .2, .3, .4):
                        s = smooth_state(prob, span, h)[pos]
                        front.append(dict(group=grp, tf=tf, asset=n, method='Model', params=f'{span},{h}',
                                          mcc=matthews_corrcoef(ev.y.values, s),
                                          flips=np.sum(s[1:] != s[:-1]) / max(np.sum(ev.y.values[1:] != ev.y.values[:-1]), 1)))
                for bn, grid in BASELINE_GRID.items():
                    for prm in grid:
                        s = np.asarray(BASELINE_FN[bn](f, prm))[pos]
                        front.append(dict(group=grp, tf=tf, asset=n, method=bn, params=str(prm),
                                          mcc=matthews_corrcoef(ev.y.values, s),
                                          flips=np.sum(s[1:] != s[:-1]) / max(np.sum(ev.y.values[1:] != ev.y.values[:-1]), 1)))
                print(grp, tf, n, len(ev), flush=True)
    pd.DataFrame(rows).to_pickle('out_stage3_rows.pkl')
    pd.DataFrame(front).to_pickle('out_stage3_frontier.pkl')


if __name__ == '__main__':
    main()

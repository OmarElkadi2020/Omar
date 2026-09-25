"""Stage 11: the frozen stage-10 STOCK model applied unchanged to crypto (it never saw crypto).
Same features; 'market' context = equal-weight index of the coin basket, same formulas as for stocks."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import build, build_bb
from .labels import oracle_labels
from .dataset import load_1h, COINS
from .stage10 import chop_features
from .evaluate import smooth_state, metrics, BASELINE_FN, price_ema

EVAL = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT', 'DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']
BPY = {'1D': 365, '4h': 2190, '1h': 8760}


def basket_market(tf):
    closes = {}
    for a in COINS:
        d = load_1h(a)
        c = d.close.resample(tf if tf != '1D' else '1D', origin='epoch').last()
        closes[a] = c
    P = pd.DataFrame(closes).sort_index()
    R = np.log(P).diff().clip(-0.4, 0.4)
    idx = np.exp(R.mean(axis=1).fillna(0).cumsum())
    m = pd.DataFrame(index=P.index)
    for n in (21, 63, 252):
        m[f'mkt_mom_{n}'] = np.log(idx / idx.shift(n))
    m['mkt_ema_dist_200'] = np.log(idx / idx.ewm(span=200, adjust=False).mean())
    s200 = P.rolling(200, min_periods=200).mean()
    m['mkt_breadth_200'] = (P > s200).sum(axis=1) / s200.notna().sum(axis=1).replace(0, np.nan)
    return m


def frame(a, tf, m):
    d = load_1h(a)
    base, X = build(d, tf)
    X = pd.concat([X, build_bb(d, tf).reindex(X.index)], axis=1)
    mk = m.reindex(X.index)
    X = pd.concat([X, mk], axis=1)
    c = base.close
    for n in (63, 252):
        X[f'rel_mom_{n}'] = np.log(c / c.shift(n)) - mk[f'mkt_mom_{n}']
    X = pd.concat([X, chop_features(base)], axis=1)
    lab, fin = oracle_labels(c.values, 1.0, return_final=True)
    f = pd.concat([base[['open', 'high', 'low', 'close']], X.astype(np.float32)], axis=1)
    f['y'] = lab
    f['final'] = np.arange(len(f)) <= fin
    return f


def main():
    fz, fzb = json.load(open('FROZEN_stage10.json')), json.load(open('FROZEN_stage10b.json'))
    p, cols = fz['params'], fz['cols']
    ms = lgb.Booster(model_file='trend_model_stocks.txt')
    o = pd.read_pickle('out_stage2_C.pkl')      # crypto model C, walk-forward (out-of-sample from 2019)
    rows = []
    for tf in ('1D', '4h', '1h'):
        m = basket_market(tf)
        for a in EVAL:
            f = frame(a, tf, m)
            pr = ms.predict(f[cols])
            S = {'STOCK model (accurate, unchanged)': smooth_state(pr, p['smooth_span'], p['hysteresis']),
                 'STOCK model (calm, unchanged)': smooth_state(pr, fzb['span'], fzb['h']),
                 'EMA200': price_ema(f, 200)}
            for bn, b in fz['baselines'].items():
                S[f'{bn} (stock-tuned)'] = np.asarray(BASELINE_FN[bn](f, b['params']))
            for bn, b in fzb['baselines'].items():
                S[f'{bn} (stock calm-tuned)'] = np.asarray(BASELINE_FN[bn](f, b['params']))
            if tf != '1D':
                x = o[(o.asset == a) & (o.tf == tf)]
                cm = pd.Series(np.nan, f.index)
                cm.loc[x.index.intersection(f.index)] = x.p.reindex(x.index.intersection(f.index)).values
                S['Crypto model C (walk-forward)'] = smooth_state(cm.values, 1, 0.204)
            for per, lo in (('all history', f.index[min(400, len(f) - 1)]), ('2019-2026', pd.Timestamp('2019-01-01', tz='UTC'))):
                w = (f.index >= lo) & f.final.values
                if w.sum() < 200 or len(set(f.y.values[w])) < 2:
                    continue
                for k, s in S.items():
                    if k.startswith('Crypto model C') and per != '2019-2026':
                        continue
                    mt = metrics(np.asarray(s)[w], f.y.values[w], f.close.values[w], fee=0.001, bars_per_year=BPY[tf])
                    mt.update(tf=tf, asset=a, clf=k, period=per)
                    rows.append(mt)
            print(tf, a, len(f), flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle('out_stage11.pkl')


if __name__ == '__main__':
    main()

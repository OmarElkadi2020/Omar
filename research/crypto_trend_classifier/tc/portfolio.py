"""Portfolio risk snapshot (descriptive, not advice). Reads holdings from a JSON file
{"COIN": {"qty": float, "cost": float}} (cost = total USD paid), merges spot 4h history (data/universe_4h) with the
latest Binance-perp hourly candles from Coinalyze (COINALYZE_API_KEY), and prints per coin and for the whole book:
weight, P&L, 4h/daily SuperTrend state and lines (stage-27 settings), 30d volatility, beta to BTC, drawdown from the
all-time high, relative strength vs BTC, liquidity, and the loss in a stress scenario (BTC to its daily line).
Also: base rates for alts vs BTC (1-year forward returns over 2019-2026, all USDT pairs in data/universe_4h).
python -m tc.portfolio <holdings.json>"""
import json
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from .intraday_snapshot import st_line

AGG = dict(open='first', high='max', low='min', close='last', volume='sum')


def coinalyze(path):
    for _ in range(8):
        try:
            r = urllib.request.Request('https://api.coinalyze.net/v1/' + path,
                                       headers={'api_key': os.environ['COINALYZE_API_KEY']})
            return json.load(urllib.request.urlopen(r, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(float(e.headers.get('Retry-After', 10)) + 1)


def history(coins, days=6):
    """4h spot history + recent perp candles (rescaled to the spot level on the overlap)."""
    now = int(time.time())
    syms = ','.join(f'{c}USDT_PERP.A' for c in coins)
    perp = {s['symbol'].split('USDT')[0]: s['history'] for s in
            coinalyze(f'ohlcv-history?symbols={syms}&interval=1hour&from={now - days * 86400}&to={now}')}
    out = {}
    for c in coins:
        sp = pd.read_parquet(f'data/universe_4h/{c}USDT.parquet').astype(float)
        sp['volume'] = sp.volume * sp.close                                   # quote volume (USD)
        p = pd.DataFrame(perp[c])
        p.index = pd.to_datetime(p.t, unit='s', utc=True)
        p = p.rename(columns=dict(o='open', h='high', l='low', c='close', v='volume'))[list(AGG)]
        p['volume'] = 0.0                                                     # perp volume not comparable to spot
        p4 = p.resample('4h', origin='epoch').agg(AGG).dropna()
        ov = sp.index.intersection(p4.index)
        k = (sp.close[ov] / p4.close[ov]).median() if len(ov) else 1.0
        p4[['open', 'high', 'low', 'close']] *= k
        out[c] = (pd.concat([sp, p4[p4.index > sp.index[-1]]]).sort_index(), float(p.close.iloc[-1] * k))
    return out


def coin_stats(d, btc_d, price):
    now = pd.Timestamp.utcnow()
    c4 = d[d.index + pd.Timedelta('4h') <= now]
    s4, l4 = st_line(c4, 48, 5.0)
    day = d.resample('1D').agg(AGG).dropna()
    c1 = day[day.index + pd.Timedelta('1D') <= now]
    s1, l1 = st_line(c1, 48, 6.0)
    r = np.log(c1.close).diff()
    rb = np.log(btc_d.close).diff().reindex(r.index)
    x = pd.concat([r, rb], axis=1).dropna().iloc[-180:]
    beta = np.cov(x.iloc[:, 0], x.iloc[:, 1])[0, 1] / x.iloc[:, 1].var()
    dn = x[x.iloc[:, 1] < 0]
    beta_dn = np.cov(dn.iloc[:, 0], dn.iloc[:, 1])[0, 1] / dn.iloc[:, 1].var()
    ret = lambda s, n: s.iloc[-1] / s.iloc[-1 - n] - 1
    bc = btc_d.close.reindex(c1.index).ffill()
    return dict(price=price, st4h='UP' if s4 == 1 else 'DOWN', line4h=float(l4), to_line4h=float(l4 / price - 1),
                st1d='UP' if s1 == 1 else 'DOWN', line1d=float(l1), to_line1d=float(l1 / price - 1),
                vol30=float(r.iloc[-30:].std() * np.sqrt(365)), beta=float(beta), beta_down=float(beta_dn),
                from_ath=float(price / c1.high.max() - 1), ath_date=str(c1.high.idxmax().date()),
                r30=float(ret(c1.close, 30)), r90=float(ret(c1.close, 90)),
                rel_btc_30=float(ret(c1.close, 30) - ret(bc, 30)), rel_btc_90=float(ret(c1.close, 90) - ret(bc, 90)),
                adv_30d_musd=float(c1.volume.iloc[-30:].mean() / 1e6))


def base_rates(start='2019-01-01', min_hist=365):
    """1-year forward return of every USDT pair vs BTC, sampled monthly; share of alts that beat BTC."""
    P = {}
    for f in os.listdir('data/universe_4h'):
        s = pd.read_parquet(f'data/universe_4h/{f}', columns=['close']).close.astype(float)
        P[f[:-12]] = s.resample('1D').last()
    P = pd.DataFrame(P)
    P.index = P.index.tz_localize(None)
    P = P.loc[start:]
    btc = P.pop('BTC')
    rows = []
    for t in pd.date_range(start, P.index[-1] - pd.Timedelta('366D'), freq='MS'):
        t1 = t + pd.Timedelta('365D')
        hist = P.loc[:t].notna().sum() >= min_hist
        a = P.loc[t, hist].dropna()
        e = P.reindex([t1]).iloc[0][a.index]
        fwd = (e / a - 1).fillna(-1.0)                                        # delisted within the year = -100%
        rows.append(dict(t=t, n=len(a), alt_median=fwd.median(), btc=btc[t1] / btc[t] - 1,
                         share_beat_btc=(fwd > btc[t1] / btc[t] - 1).mean(), share_double=(fwd >= 1).mean()))
    return pd.DataFrame(rows).set_index('t')


def main():
    H = json.load(open(sys.argv[1]))
    coins = list(H)
    data = history(coins + ([] if 'BTC' in H else ['BTC']))
    btc_day = data['BTC'][0].resample('1D').agg(AGG).dropna()
    rows = []
    for c in coins:
        d, px = data[c]
        s = coin_stats(d, btc_day, px)
        v = H[c]['qty'] * px
        rows.append(dict(coin=c, value=v, cost=H[c]['cost'], pnl=v / H[c]['cost'] - 1, **s))
    T = pd.DataFrame(rows).set_index('coin')
    T['weight'] = T.value / T.value.sum()
    btc_px = data['BTC'][1]
    btc_line = coin_stats(data['BTC'][0], btc_day, btc_px)['line1d']
    shock = btc_line / btc_px - 1
    T['stress_btc_to_daily_line'] = T.value * T.beta_down * shock
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(T.round(3).to_string())
    tot = dict(value=T.value.sum(), cost=T.cost.sum(), pnl=T.value.sum() / T.cost.sum() - 1,
               breakeven_needed=T.cost.sum() / T.value.sum() - 1, top1=T.weight.max(),
               in_4h_down=T.weight[T.st4h == 'DOWN'].sum(), in_1d_down=T.weight[T.st1d == 'DOWN'].sum(),
               beta=(T.weight * T.beta).sum(), beta_down=(T.weight * T.beta_down).sum(),
               btc_price=btc_px, btc_daily_line=btc_line, btc_shock=shock,
               stress_loss=T.stress_btc_to_daily_line.sum(), stress_loss_pct=T.stress_btc_to_daily_line.sum() / T.value.sum())
    print({k: round(v, 3) for k, v in tot.items()})
    B = base_rates()
    print('\nalts vs BTC, 1-year forward (monthly starts 2019 ->):')
    print(B[['n', 'alt_median', 'btc', 'share_beat_btc', 'share_double']].describe().round(3).to_string())
    print('years:', B.groupby(B.index.year)[['alt_median', 'btc', 'share_beat_btc']].mean().round(3).to_string())
    out = sys.argv[1].replace('.json', '_snapshot.json')
    json.dump(dict(time=str(pd.Timestamp.utcnow().floor('min')), coins=T.round(4).reset_index().to_dict('records'),
                   total=tot, base_rates=B.describe().round(4).to_dict()), open(out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()

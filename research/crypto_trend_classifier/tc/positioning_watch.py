"""Positioning / flow watch on Binance perps (descriptive, not advice). Needs COINALYZE_API_KEY.
For BTC, ETH, LINK, XRP and windows of 1h, 4h, 24h (completed hours only): price change, open-interest change
(in coins), taker-buy share and net taker flow (aggressive buys minus sells, USD), long/short liquidations (USD),
the latest funding rate and long/short account ratio, and the usual reading of price x OI:
price up + OI up = new longs, price up + OI down = short covering, price down + OI up = new shorts,
price down + OI down = longs closing / liquidated. The reading is a convention, not proof of who traded.
Prints one JSON line and appends it to data/positioning_log.jsonl (gitignored).
python -m tc.positioning_watch"""
import json
import os
import time
import pandas as pd
from .intraday_snapshot import get

COINS = ('BTC', 'ETH', 'LINK', 'XRP')
LOG = 'data/positioning_log.jsonl'


def fetch(days=3):
    now = int(time.time())
    fr = now - days * 86400
    syms = ','.join(f'{c}USDT_PERP.A' for c in COINS)
    out = {c: {} for c in COINS}
    for ep, extra, fields in (('ohlcv-history', '', ('c', 'v', 'bv')), ('open-interest-history', '', ('c',)),
                              ('liquidation-history', '&convert_to_usd=true', ('l', 's')),
                              ('funding-rate-history', '', ('c',)), ('long-short-ratio-history', '', ('r',))):
        for s in get(f'{ep}?symbols={syms}&interval=1hour&from={fr}&to={now}{extra}'):
            d = pd.DataFrame(s['history'])
            d.index = pd.to_datetime(d.t, unit='s', utc=True)
            c = s['symbol'].split('USDT')[0]
            for f in fields:
                out[c][f'{ep.split("-")[0]}_{f}'] = d[f]
        time.sleep(2)
    hour = pd.Timestamp.utcnow().floor('h')
    return {c: pd.DataFrame(v).loc[lambda x: x.index < hour] for c, v in out.items()}      # completed hours only


def reading(dp, doi):
    if abs(dp) < 0.003 and abs(doi) < 0.003:
        return 'flat'
    return {(True, True): 'new longs', (True, False): 'short covering',
            (False, True): 'new shorts', (False, False): 'longs closing/liquidated'}[(dp >= 0, doi >= 0)]


def window(d, n):
    w = d.iloc[-n:]
    p0, p1 = d.ohlcv_c.iloc[-n - 1], d.ohlcv_c.iloc[-1]
    o0, o1 = d.open_c.iloc[-n - 1], d.open_c.iloc[-1]
    dp, doi = p1 / p0 - 1, o1 / o0 - 1
    v, bv = w.ohlcv_v.sum(), w.ohlcv_bv.sum()
    return dict(price_pct=round(dp * 100, 2), oi_pct=round(doi * 100, 2), read=reading(dp, doi),
                taker_buy_share=round(bv / v, 3) if v else None,
                net_taker_usd_m=round((2 * bv - v) * w.ohlcv_c.mean() / 1e6, 1),
                liq_long_usd_m=round(w.liquidation_l.sum() / 1e6, 1), liq_short_usd_m=round(w.liquidation_s.sum() / 1e6, 1))


def main():
    D = fetch()
    out = dict(time=str(pd.Timestamp.utcnow().floor('min')), last_hour=str(D['BTC'].index[-1]))
    for c, d in D.items():
        out[c] = dict(price=float(d.ohlcv_c.iloc[-1]), oi_coins=float(d.open_c.iloc[-1]),
                      funding_pct=round(float(d.funding_c.dropna().iloc[-1]), 4),
                      ls_ratio=round(float(d.long_r.dropna().iloc[-1]), 2) if 'long_r' in d and d.long_r.notna().any() else None,
                      **{f'w{n}h': window(d, n) for n in (1, 4, 24)})
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(out, default=str) + '\n')
    print(json.dumps(out, default=str, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

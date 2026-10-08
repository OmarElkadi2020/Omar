"""Macro / flows watch (descriptive, not advice): the US dollar, Treasury yields, oil, gold, S&P 500, the Fed balance
sheet and US net liquidity, US spot-ETF flows for BTC/ETH, and Fear & Greed. Prints one JSON line and appends it to
data/macro_log.jsonl (gitignored). Keys from env: FMP_API_KEY (FX, Brent, gold, S&P, Treasury curve), FRED_API_KEY
(balance sheet, TGA, RRP), BYkaranteli_API_KEY (ETF flows). Every source is optional: a failed source is reported as
null rather than stopping the run.
DXY is rebuilt from the ICE formula with live FX quotes. USD/SEK is not on the FMP plan, so SEK is proxied as
EUR/SEK = 11.0 (SEK weight 4.2%; a 5% EUR/SEK move shifts the index by about 0.2%).
python -m tc.macro_watch"""
import json
import os
import urllib.request
import pandas as pd

LOG = 'data/macro_log.jsonl'
DXY_W = {'EURUSD': -0.576, 'USDJPY': 0.136, 'GBPUSD': -0.119, 'USDCAD': 0.091, 'USDCHF': 0.036}
EURSEK = 11.0


def fmp(path):
    u = f'https://financialmodelingprep.com/stable/{path}{"&" if "?" in path else "?"}apikey={os.environ["FMP_API_KEY"]}'
    return json.load(urllib.request.urlopen(u, timeout=30))


def quotes():
    q = {}
    for s in list(DXY_W) + ['BZUSD', 'GCUSD', '^GSPC']:
        d = fmp(f'quote?symbol={s}')[0]
        q[s] = (d['price'], d['previousClose'])
    def dxy(i):
        v = 50.14348112 * (EURSEK / q['EURUSD'][i]) ** 0.042
        for s, w in DXY_W.items():
            v *= q[s][i] ** w
        return v
    pct = lambda s: round((q[s][0] / q[s][1] - 1) * 100, 2)
    return dict(dxy=round(dxy(0), 2), dxy_chg_pct=round((dxy(0) / dxy(1) - 1) * 100, 2),
                usdjpy=q['USDJPY'][0], eurusd=q['EURUSD'][0],
                brent=q['BZUSD'][0], brent_chg_pct=pct('BZUSD'), gold=q['GCUSD'][0], gold_chg_pct=pct('GCUSD'),
                spx=q['^GSPC'][0], spx_chg_pct=pct('^GSPC'))


def treasuries():
    r = fmp('treasury-rates')[:6]
    a, b, w = r[0], r[1], r[5]
    return dict(date=a['date'], y2=a['year2'], y10=a['year10'], y30=a['year30'], m3=a['month3'],
                y10_chg_1d_bp=round((a['year10'] - b['year10']) * 100), y10_chg_1w_bp=round((a['year10'] - w['year10']) * 100),
                curve_2s10s_bp=round((a['year10'] - a['year2']) * 100))


def fred(series, start):
    u = (f'https://api.stlouisfed.org/fred/series/observations?series_id={series}&observation_start={start}'
         f'&file_type=json&api_key={os.environ["FRED_API_KEY"]}')
    o = json.load(urllib.request.urlopen(u, timeout=30))['observations']
    return pd.Series({pd.Timestamp(x['date']): float(x['value']) for x in o if x['value'] != '.'}).sort_index()


def liquidity():
    st = str((pd.Timestamp.utcnow() - pd.Timedelta('200D')).date())
    walcl = fred('WALCL', st) / 1e3
    nl = (walcl - fred('WTREGEN', st).reindex(walcl.index, method='ffill') / 1e3
          - fred('RRPONTSYD', st).reindex(walcl.index, method='ffill')).dropna()
    return dict(date=str(walcl.index[-1].date()), fed_assets_bn=round(walcl.iloc[-1]),
                fed_assets_4w_bn=round(walcl.iloc[-1] - walcl.iloc[-5]), fed_assets_13w_bn=round(walcl.iloc[-1] - walcl.iloc[-14]),
                net_liq_bn=round(nl.iloc[-1]), net_liq_4w_bn=round(nl.iloc[-1] - nl.iloc[-5]))


def etf():
    from .download_bykaranteli import fetch, merge
    f = merge('etf-flows', fetch('etf-flows'))
    out = {}
    for a in ('BTC', 'ETH'):
        x = f[f.asset == a].sort_values('date')
        fl = x.net_inflow_usd / 1e6
        out[a] = dict(last_date=str(x.date.iloc[-1])[:10], last_m=round(fl.iloc[-1], 1), d5_m=round(fl.tail(5).sum(), 1),
                      d20_m=round(fl.tail(20).sum(), 1),
                      outflow_streak_days=next((i for i, v in enumerate(fl[::-1]) if v >= 0), len(fl)))
    return out


def fear_greed():
    d = json.load(urllib.request.urlopen('https://api.alternative.me/fng/?limit=2', timeout=30))['data']
    return dict(now=int(d[0]['value']), label=d[0]['value_classification'], prev=int(d[1]['value']))


def main():
    out = dict(time_berlin=pd.Timestamp.utcnow().tz_convert('Europe/Berlin').strftime('%Y-%m-%d %H:%M %Z'), time=str(pd.Timestamp.utcnow().floor('min')))
    for k, f in dict(markets=quotes, treasuries=treasuries, liquidity=liquidity, etf=etf, fear_greed=fear_greed).items():
        try:
            out[k] = f()
        except Exception as e:
            out[k] = None
            out.setdefault('errors', {})[k] = f'{type(e).__name__}: {e}'[:200]
    prev = None
    if os.path.exists(LOG):
        lines = [x for x in open(LOG) if x.strip()]
        prev = json.loads(lines[-1]) if lines else None
    if prev and prev.get('markets') and out.get('markets'):
        out['since_last'] = dict(prev_time=prev['time'],
                                 dxy=round(out['markets']['dxy'] - prev['markets']['dxy'], 2),
                                 brent=round(out['markets']['brent'] - prev['markets']['brent'], 2))
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(out, default=str) + '\n')
    print(json.dumps(out, default=str, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

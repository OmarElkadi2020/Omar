"""Market context for the live assessment (information only; stages 25, 26, 32 found no money in these signals).
Needs FRED_API_KEY, COIN_MARKET_CAP_API_KEY.  python -m tc.market_context"""
import json
import os
import urllib.request
import pandas as pd


def fred(series, start='2025-01-01'):
    u = (f'https://api.stlouisfed.org/fred/series/observations?series_id={series}&observation_start={start}'
         f'&file_type=json&api_key={os.environ["FRED_API_KEY"]}')
    o = json.load(urllib.request.urlopen(u, timeout=60))['observations']
    s = pd.Series({pd.Timestamp(x['date']): float(x['value']) for x in o if x['value'] != '.'})
    return s.sort_index()


def cmc_global():
    r = urllib.request.Request('https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest',
                               headers={'X-CMC_PRO_API_KEY': os.environ['COIN_MARKET_CAP_API_KEY']})
    return json.load(urllib.request.urlopen(r, timeout=60))['data']


def main():
    walcl = fred('WALCL') / 1e3          # millions -> billions
    tga = fred('WTREGEN') / 1e3          # millions -> billions
    rrp = fred('RRPONTSYD')              # billions
    nl = (walcl - tga.reindex(walcl.index, method='ffill') - rrp.reindex(walcl.index, method='ffill')).dropna()
    g = cmc_global()
    q = g['quote']['USD']
    out = dict(
        us_net_liquidity_bn=round(float(nl.iloc[-1]), 1), us_net_liquidity_date=str(nl.index[-1].date()),
        us_net_liquidity_4w_change_bn=round(float(nl.iloc[-1] - nl.iloc[-5]), 1),
        us_net_liquidity_13w_change_bn=round(float(nl.iloc[-1] - nl.iloc[-14]), 1),
        fed_funds_upper=fred('DFEDTARU').iloc[-1], us10y=fred('DGS10').iloc[-1],
        btc_dominance=round(g['btc_dominance'], 2), eth_dominance=round(g['eth_dominance'], 2),
        total_mcap_bn=round(q['total_market_cap'] / 1e9, 1),
        total_mcap_24h_pct=round(q.get('total_market_cap_yesterday_percentage_change', float('nan')), 2),
        stablecoin_mcap_bn=round(q.get('stablecoin_market_cap', float('nan')) / 1e9, 1),
        stablecoin_24h_pct=round(q.get('stablecoin_24h_percentage_change', float('nan')), 2),
        cmc_last_updated=g.get('last_updated'))
    json.dump(out, open('results_market_context.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()

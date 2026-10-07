"""Long-term BTC picture (descriptive): halving-cycle comparison, weekly swing structure since the Oct-2025 peak,
long moving averages, weekly SuperTrend, and US spot-ETF flows (data/bykaranteli, see tc/download_bykaranteli.py).
Reproduces MARKET_LONGTERM_2026-10-07.md.   python -m tc.longterm_state"""
import numpy as np
import pandas as pd
from .evaluate import supertrend

CYCLES = {'2017': ('2017-12-01', '2018-01-31', '2019-02-28'), '2021': ('2021-10-15', '2021-11-30', '2023-01-31'),
          '2025': ('2025-09-15', '2025-11-15', '2100-01-01')}


def btc():
    d = pd.read_parquet('data/universe_4h/BTCUSDT.parquet').astype(float)
    d.index = d.index.tz_localize(None)
    agg = dict(open='first', high='max', low='min', close='last')
    return d.resample('1D').agg(agg).dropna(), d.resample('W-SUN').agg(agg).dropna()


def cycles(D):
    c, rows = D.close, []
    for name, (a, b, end) in CYCLES.items():
        pt = D.high.loc[a:b].idxmax()
        pv = D.high[pt]
        lw = D.low.loc[pt:end]
        bt = lw.idxmin()
        seg = D.loc[pt:bt]
        best = max(seg.high.values[i:].max() / seg.low.values[i] - 1 for i in range(len(seg)))
        path = {f'{m}m': c.asof(pt + pd.Timedelta(days=int(m * 30.4))) / pv - 1
                for m in (3, 6, 9, 12, 15, 18) if pt + pd.Timedelta(days=int(m * 30.4)) <= c.index[-1]}
        rows.append(dict(cycle=name, peak=pv, peak_date=pt.date(), low=lw.min(), low_date=bt.date(),
                         drawdown=lw.min() / pv - 1, days_to_low=(bt - pt).days, biggest_rally_in_decline=best,
                         rally_since_low=D.high.loc[bt:].max() / lw.min() - 1, **path))
    return pd.DataFrame(rows).set_index('cycle')


def swings(W, since='2025-09-01', k=3):
    w, out = W.loc[since:], []
    for i in range(k, len(w) - k):
        if w.high.iloc[i] == w.high.iloc[i - k:i + k + 1].max():
            out.append(('HIGH', w.index[i].date(), round(w.high.iloc[i])))
        if w.low.iloc[i] == w.low.iloc[i - k:i + k + 1].min():
            out.append(('LOW', w.index[i].date(), round(w.low.iloc[i])))
    return out


def trend_levels(D, W):
    c = D.close
    s200w, s50w = W.close.rolling(200).mean(), W.close.rolling(50).mean()
    return dict(close=c.iloc[-1], sma_200w=s200w.iloc[-1], sma_50w=s50w.iloc[-1],
                sma_50w_slope_8w=s50w.iloc[-1] / s50w.iloc[-9] - 1, sma_365d=c.rolling(365).mean().iloc[-1],
                weekly_st_10_3='UP' if supertrend(W, 10, 3.0)[-1] == 1 else 'DOWN',
                weekly_st_20_4='UP' if supertrend(W, 20, 4.0)[-1] == 1 else 'DOWN')


def etf(asset='BTC', wave_start='2026-08-20'):
    F = pd.read_csv('data/bykaranteli/etf-flows.csv', parse_dates=['date'])
    f = F[F.asset == asset].set_index('date').sort_index()
    fl = f.net_inflow_usd / 1e6
    sym = f'{asset}USDT'
    p = pd.read_parquet(f'data/universe_4h/{sym}.parquet').close.astype(float).resample('1D').last()
    p.index = p.index.tz_localize(None)
    r = np.log(p).diff().reindex(fl.index)
    corr = lambda x, y: pd.concat([x, y], axis=1).dropna().corr().iloc[0, 1]
    return dict(first=str(f.index[0].date()), last=str(f.index[-1].date()),
                net_assets_first_bn=f.net_assets_usd.iloc[0] / 1e9, net_assets_last_bn=f.net_assets_usd.iloc[-1] / 1e9,
                flow_total_m=fl.sum(), flow_since_wave_m=fl.loc[wave_start:].sum(), flow_20d_m=fl[-20:].sum(),
                flow_5d_m=fl[-5:].sum(), flow_last_m=fl.iloc[-1],
                monthly_m=fl.resample('ME').sum().round(0).astype(int).rename(lambda t: t.strftime('%Y-%m')).to_dict(),
                corr_same_day=corr(fl, r), corr_prev_day_return=corr(fl, r.shift(1)),
                corr_next_day_return=corr(fl, np.log(p.shift(-1) / p).reindex(fl.index)),
                corr_5d_flow_next_week=corr(fl.rolling(5).sum(), np.log(p.shift(-7) / p).reindex(fl.index)))


def main():
    pd.set_option('display.width', 220)
    D, W = btc()
    print('data to', D.index[-1].date())
    print(cycles(D).round(3).T.to_string())
    print('\nweekly swing points:', swings(W))
    print('\n', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in trend_levels(D, W).items()})
    for a in ('BTC', 'ETH'):
        print(f'\n{a} ETFs:', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in etf(a).items()})


if __name__ == '__main__':
    main()

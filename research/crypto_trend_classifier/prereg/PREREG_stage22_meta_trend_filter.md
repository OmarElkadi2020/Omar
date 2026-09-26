# Stage 22 pre-registration: a meta trend filter that must beat every classic trend indicator

Written and committed before any stage-22 feature, model or return is computed.

## Lessons carried over (stages 12-21)
* A 7-day up/down target gives a short, relative signal (stage 21: ON state did not separate 30-day returns).
  -> target is the **30-day** forward return (vol-normalised), the horizon at which EMA200 separated best.
* The model must see the classic indicators themselves -> features = their states/distances on **daily and
  weekly** bars (MTF), plus volatility regime and market context; it learns when to trust which.
* It must also beat a no-ML ensemble -> benchmark VOTE (ON when ≥ 5 of the 9 indicators are ON).
* The crypto 2022-26 window has been looked at in stages 15, 16, 19, 20, 21 -> a second, **fresh** test is added:
  per-stock trend filtering on Indian NSE stocks (only cross-sectional work was done there before).

## Features (per asset, time-series, scale-free; daily bar t uses data ≤ t; weekly bar = last completed week)
Daily: close/EMA-1 and EMA slope for 20/50/100/200 (in σ units); SMA50>SMA200; MACD hist/σ and MACD>signal;
TSMOM 10/30/60/90/180/365 (vol-normalised); Donchian position 20/55/100 and Donchian 20/10 and 55/20 states;
SuperTrend(10,3) and (20,4) state and distance; ADX14, (+DI − −DI)/(+DI + −DI); RSI14; efficiency ratio 30/90;
choppiness 14; trend R² 60/180; vol 30 / vol 365; drawdown from 365-day high; days since the last EMA50 and
EMA200 cross; number of the 9 benchmark indicators that are ON. Weekly: close/EMA10w, close/EMA40w, EMA10w>EMA40w,
weekly RSI14, 13- and 52-week return, weekly SuperTrend(10,3). Context: the same trend features of the market
proxy (crypto: BTC; India: equal-weight universe) + breadth (share of the universe above EMA50 / EMA200).

## Target and model
`y = clip(log(P_{t+30}/P_t) / (σ30_t·√30), −4, 4)` from the execution price. LightGBM regression, pooled over
every asset of the market's training universe (crypto: all Binance USDT pairs incl. delisted, top-100 liquid;
India: top-500). Annual expanding walk-forward, purge 30 + 7 days. Filter: `s = EWM_span(pred)`,
ON when s > δ_in, OFF when s < δ_out (hysteresis).

## Dev (tuning) — per market, only pre-test years
Crypto folds 2019, 2020, 2021 (evaluated on the 10 large caps that exist); India folds 2006 … 2010
(evaluated on the top-100 by traded value). Optuna 25 trials seed 0 over LightGBM params + span ∈ {1,5,10,20}
+ δ_in ∈ {−0.1, 0, 0.1} + δ_out ∈ {−0.2, −0.1, 0} (δ_out ≤ δ_in). Objective = mean fold α t-stat (below).

## Test portfolios (same construction for every filter)
* A — crypto: BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX, 10 % slots, cash when OFF, next-4h execution, 10 bp,
  2022-01-01 … 2026-08-31.
* B — India: each day the top-100 by traded value, equal slots (1/100), cash when OFF, next-open execution,
  15 bp, 2011-01-01 … 2026-09-24.
Benchmarks: buy & hold, EMA50, EMA200, golden cross, MACD, TSMOM-30, TSMOM-90, Donchian 20/10,
SuperTrend(10,3), ADX>25 with +DI>−DI, VOTE (≥ 5 of 9).

## Primary endpoint (goal met only if A and B both pass)
For each test: α > 0 with NW (lag 20) t ≥ 2.5 on all benchmark portfolios at once **and** Sharpe above
every benchmark.

## Secondary
30-day forward return ON vs OFF; share of > 20 % 30-day crashes avoided and rallies captured; flips per
asset-year; time in market; per-asset Sharpe wins; per-year returns.

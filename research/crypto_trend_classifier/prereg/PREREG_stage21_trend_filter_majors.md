# Stage 21 pre-registration: the model as a long-only TREND FILTER on large-cap crypto

Written and committed before any stage-21 number is computed.

## Use case (user)
The indicator is used only as a trend filter (long entries allowed / not allowed); entries come from other
indicators. So the filter is judged by what it does to exposure: be in during up-trends, out during
down-trends, with few flips.

## Assets (fixed now)
BTC, ETH, BNB, SOL, XRP, ADA, DOGE, LINK, AVAX, TRX (USDT spot, Binance). Daily decisions at the daily close,
using daily + 4h (MTF) inputs; execution at the next 4h close; 10 bp per unit turnover; cash earns 0.

## Filter
The frozen stage-20 model (trained on every Binance USDT coin incl. delisted, annual walk-forward, no change)
gives p(up over 7 days) per coin per day. Filter state per coin with hysteresis:
`s = EWM_span(p)`; ON when s > θ_in, OFF when s < θ_out (θ_out ≤ θ_in), else keep the previous state.
Dev-tuned on 2019, 2020, 2021 (out-of-fold p from models trained strictly before each year):
span ∈ {1, 3, 7, 14}, θ_in ∈ {0.50, 0.55, 0.60}, θ_out ∈ {0.40, 0.45, 0.50} with θ_out ≤ θ_in (full grid,
objective = mean over the 3 years of the α t-stat defined below).

## Benchmark filters (same coins, execution, costs)
EMA50, EMA200, golden cross (SMA50 > SMA200), MACD(12,26,9) > signal, TSMOM-30, TSMOM-90, Donchian 20/10,
SuperTrend(10,3), ADX(14) > 25 & +DI > −DI, and buy & hold (always ON).

## Portfolio used for the test
Each coin is a fixed 10 % slot; the slot is in the coin when its filter is ON, else cash (coins not yet listed
= cash for every strategy). Same construction for every filter.

## Primary endpoint — test 2022-01-01 … 2026-08-31 (crypto window used before; hurdle stays strict)
1. α > 0 with NW (lag 20) t ≥ 2.5 of the model-filter portfolio on all benchmark-filter portfolios + buy & hold.
2. Model-filter Sharpe above every benchmark filter's Sharpe.

## Secondary (filter quality, per coin and pooled)
Mean forward 7- and 30-day return when ON vs when OFF; share of the worst 30-day drawdowns avoided
(OFF at the start of a 30-day window that falls > 20 %); share of 30-day rallies > 20 % captured
(ON at the start); flips per coin per year; time in market; per-coin Sharpe wins vs each benchmark.

## Amendment 1 (dev only, before any test number)
On the dev folds the raw p of the 2019/2020 models never crossed 0.5 (models trained on the 2018 bear market
are shifted down; magnitude-weighted binary loss is not calibrated), so the filter was never ON. Change:
each yearly model's own base rate b_Y = mean predicted p on its training rows (known at training time) is
subtracted: `s = EWM_span(p − b_Y)`, ON when s > δ_in, OFF when s < δ_out. Grid: span ∈ {1,3,7,14},
δ_in ∈ {−0.05, 0, 0.05}, δ_out ∈ {−0.10, −0.05, 0} with δ_out ≤ δ_in. Everything else unchanged. The test
models are re-fitted with the frozen stage-20 params (deterministic, same p) to obtain their b_Y.

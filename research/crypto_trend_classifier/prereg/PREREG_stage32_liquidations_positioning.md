# Stage 32 pre-registration: do liquidations and positioning improve the best simple rule?

Written and committed before any Coinalyze history is downloaded. Only data availability was checked
(BTCUSDT_PERP.A daily: liquidations and open interest start 2022-08-27, funding 2020-01-21, long/short ratio
2020-05-31).

## Question
The uploaded "institutional framework" (and common practice) says (a) buy after a big long-liquidation flush and
(b) step aside when longs are crowded (high funding). Neither was ever tested here. Do they add money on top of the
best simple rule we have?

## Base rule (A, unchanged)
SuperTrend(48, 5) on 4h, long/flat, × stage-27 vol sizing (2 % daily-vol target from trailing 30-day std of 4h
returns, cap 2×) — `stage30b.vol_lev`. 10 coins (BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX), equal weight,
Binance spot 4h bars from `data/binance_1h`, cost 0.1 % per side on |Δposition|.

## Data (Coinalyze, key `COINALYZE_API_KEY`; downloader `tc/download_coinalyze.py`, committed in `data/coinalyze/`)
Daily, Binance USDT perpetual of each coin (`<COIN>USDT_PERP.A`): long and short liquidations (USD), open interest
(close, USD), funding rate (close). A daily value for day d is used only from d+1 00:00 UTC (one full day lag).

## Hypotheses (each is an overlay on A; everything else stays A)
* **H1 flush-buy:** flush on day d if long liquidations / open interest is above its trailing 90-day mean by
  more than z* trailing standard deviations. For K days after a flush the coin is held long with the vol-sized
  position even if SuperTrend is down (the dip buy); otherwise A.
  Grid: z* ∈ {2, 3, 4}, K ∈ {3, 7, 14}.
* **H2 crowded exit:** crowded on day d if the funding rate is above its trailing 365-day q-quantile. While crowded,
  position = s × A. Grid: q ∈ {0.80, 0.90, 0.95}, s ∈ {0, 0.5}.

## Dev / test
* H1 dev: 2022-12-01 … 2023-12-31 (90-day warm-up after the data start). H2 dev: 2021-01-01 … 2023-12-31.
* Choice per hypothesis by dev portfolio Sharpe (one choice each, frozen in `FROZEN_stage32.json`).
* **Test (run once): 2024-01-01 … 2026-10-01**, same for both.

## Primary endpoint (money; Bonferroni over 2 hypotheses)
For each hypothesis: Sharpe(A + H) − Sharpe(A) > 0 with paired stationary block bootstrap (daily returns, block 20,
10,000) one-sided p < 0.025, and A + H better in ≥ 6 of 10 coins. Reported separately; a hypothesis "works" only if
both hold.

## Secondary
CAGR, max DD, turnover, number of flush / crowded days, per year, per coin; the same overlays on plain SuperTrend;
long-liquidation spikes vs short-liquidation spikes (descriptive).

## Power, said in advance
Overlays act on few days (flushes are rare), so the money difference will be small and noisy; p < 0.025 needs a
large effect. A "not met" mostly means "no large effect".

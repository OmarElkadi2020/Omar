# Stage 29 pre-registration: profit-weighted training, judged in money

Written and committed before any stage-29 feature, model or result is computed.

## Question
The B1 label (oracle, k = 1) gives every 4h bar the same weight, so a wrong side on a 0.1 % bar costs the model as much
as a wrong side on a 10 % bar. In money the second costs 100× more. **If each training bar is weighted by the size
of the move that follows it, does the model make more money than the same model trained without weights?**
From this stage on, every comparison is in money and trades (after costs), not MCC or AUC.

## Data (the only price data reachable in this environment)
4h OHLC embedded in `trend_compass.html`: BTC ETH BNB XRP SOL (USDT), 2019-01-01 … 2026-09-23 20:00 UTC
(SOL from 2020-08-11). No volume, so features are price-only. Stage 25 found price-only ≈ all blocks
(MCC 0.374 vs 0.387), so this costs little. Exporter: `tc/stage29.py data`.

## Features
`features.core_features` on 4h bars, plus the `HTF_KEEP` set from 16h (×4) and 4-day (×24) candles joined by close
time (the `features.build` recipe). A truncation leak test (features recomputed on data cut at 6 dates, compared bar
by bar) must return 0 mismatches before training.

## Label and training rows
B1: `oracle_labels(close, k = 1)` on 4h closes. Training uses only labels final at the cut-off, minus a 24-bar
embargo (as stage 25). Pooled over the 5 coins. Refit once a year on everything before 1 January.

## Models (identical except the weights)
LightGBM with the frozen stage-25 parameters (leaves 11, lr 0.01097, 150 trees, min leaf 525, ff 0.431, bf 0.940,
l2 75.2). No hyper-parameter tuning in this stage.
* **EQUAL:** no weights (stage-25 style).
* **WEIGHTED (primary):** weight of bar t = |log(c[t+1] / c[t])|, the money a wrong side on that bar costs per unit
  of position. Weights are scaled to mean 1 in each training set. The next return of a training row is always before
  the cut-off (rows end ≥ 24 bars before it).
* **WEIGHTED-24h (secondary):** weight = |log(c[t+6] / c[t])|, the move over the next 24 h.

## From probability to position
Score = EWM(2p − 1, span), then a binary switch at ±θ (stage-25 `model_state`). Grid: span {1, 3, 6, 12, 24} ×
θ {0, 0.1, 0.2, 0.3, 0.4}. **Chosen per model on dev by money**: mean over dev folds 2022 and 2023 of the portfolio
Sharpe (long/flat, 0.1 % cost). Each fold's model is trained only on data before that fold.

## Positions, costs, portfolio
* Primary: **long/flat** (the user's use: trend filter for buying majors). Secondary: long/short.
* Cost 0.1 % per side on every change of position. Stress test: 0.2 %.
* Portfolio: 1/5 of capital per coin (equal weight, rebalanced every bar). Before SOL exists, the 4 others.

## Benchmarks (no tuning)
SuperTrend(48, 5) on 4h (frozen stage-27 choice); equal-weight buy & hold; the B1 label itself (hindsight ceiling).

## Test (run once)
2024-01-01 … 2026-09-23. Models refit on 1 Jan 2024, 2025, 2026. The period was used by stages 25-27 for other
questions on crypto 4h; this weighted-vs-equal comparison has never been run on it.

## Money evaluation (the project standard from now on)
Per strategy, on the portfolio and per coin:
1. **Money:** CAGR, annual volatility, Sharpe, max drawdown, Calmar, time in market.
2. **Trades:** trades per coin-year, win rate, average win / average loss, profit factor, expectancy per trade,
   share of total P&L made by the best 10 % of trades.
3. **Label vs prediction, in money:**
   * **Capture** = P&L(strategy) / P&L(B1 label traded the same way, same costs): the share of the perfect-hindsight
     profit the strategy keeps.
   * **Regret split** (log-return per year), which sums exactly to P&L(label) − P&L(strategy):
     (a) *missed moves*: bars where the label is long and the strategy is not;
     (b) *wrong-side bars*: bars where the strategy is long and the label is not;
     (c) *extra costs*: strategy costs − label costs.
   * **Money-weighted accuracy** = Σ|r|·[strategy = label] / Σ|r|: accuracy where each bar counts by its move.
     Reported next to plain accuracy.
4. **Statistics:**
   * Sharpe difference between two strategies: paired stationary block bootstrap (Politis-Romano) on daily
     portfolio returns, mean block 20 days, 10,000 resamples; one-sided p-value.
   * Alpha vs buy & hold: OLS of daily returns on buy-and-hold daily returns, Newey-West t (10 lags).

## Primary endpoint (test, long/flat, 0.1 % cost; all three must hold)
1. Sharpe(WEIGHTED) − Sharpe(EQUAL) > 0 with bootstrap one-sided p < 0.05.
2. Capture(WEIGHTED) > Capture(EQUAL).
3. WEIGHTED has the higher Sharpe in ≥ 4 of 5 coins.

## Secondary
WEIGHTED vs SuperTrend(48, 5) with the same test (a "beats SuperTrend" claim needs p < 0.05); long/short; 0.2 %
cost; WEIGHTED-24h; per year; per coin.

## Power, said in advance
2.7 test years on 5 correlated coins (≈ 1.4 independent coins, stage-26 diagnostic). Only a Sharpe gap of roughly
0.6 or more can reach p < 0.05. A smaller real gain is likely to read "not met".

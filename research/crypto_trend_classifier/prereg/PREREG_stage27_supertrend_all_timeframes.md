# Stage 27 pre-registration: robust best SuperTrend settings on every timeframe (crypto), plus volatility sizing

Written and committed before the 15m data are processed and before any stage-27 number is computed.
(1h and 4h SuperTrend results for a few settings are known from stages 25-26; the grid below is far wider.)

## Question
For each timeframe, which SuperTrend(period, multiplier) is best for large-cap crypto **in a way that survives
out of sample** (not the lucky argmax of one period)? And does inverse-volatility sizing (stage-26 study: exploratory
Sharpe 0.60 → 0.87) hold up as a pre-registered test on every timeframe?

## Data
Binance spot, 20 coins: BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX (evaluation) + DOT LTC BCH ATOM NEAR UNI FIL ETC
XLM AAVE (dev only, and reported on test). Timeframes (UTC bars): **15m, 30m** (from 15m klines) and **1h, 2h, 4h, 8h,
12h, 1d** (from 1h klines).

## Grid (108 settings per timeframe)
period ∈ {5, 7, 10, 14, 20, 24, 30, 40, 48, 60, 80, 100} × multiplier ∈ {1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6}.
No other parameter. Indicator computed on the full history (it is causal; nothing is trained).

## Periods
Dev: calendar years **2019, 2020, 2021, 2022, 2023** as five folds, all 20 coins.
Test (run once): **2024-01-01 … 2026-08-31**, 10 evaluation coins (the 10 extra coins reported as secondary).

## Two scores per setting (per fold, then averaged over folds)
* **ACC** (trend classification, the project standard): mean MCC vs the oracle trend on that timeframe's closes
  (`oracle_labels`, k = 1) − 0.1 · max(0, median flips-per-true-flip − 2). The last 60 bars of the test window are
  not evaluated.
* **FILTER** (long filter quality): Sharpe of an equal-weight long-only portfolio, position = SuperTrend up-state known
  at the bar close, earning the next bar's return, 10 bp per position change.

## Selection rule (anti-overfitting)
Each dev score is **smoothed over the 3 × 3 neighbourhood of the grid** (mean of the available neighbours). The choice
is the argmax of the smoothed score (a plateau centre, not a spike). Ties → larger period. This gives an ACC choice and a
FILTER choice per timeframe. Reported but not used: the raw argmax and each dev year's argmax (stability).

## Primary endpoints (test; each evaluated separately)
A. **ACC choice is robust:** its test ACC is ≥ the 75th percentile of the 108 settings on test in ≥ 6 of 8
   timeframes, **and** it beats the default SuperTrend(10, 3) on test ACC in ≥ 6 of 8 timeframes.
B. **FILTER choice is robust:** the same two conditions with test FILTER Sharpe.
C. **Volatility sizing:** position = state × min(σ* / σ̂, 2), σ̂ = trailing 30-day standard deviation of bar
   returns (known at the bar close), σ* = 2 % daily volatility scaled to the bar length (fixed, not tuned). For the
   FILTER choice on each timeframe: Sharpe(sized) > Sharpe(equal) in ≥ 6 of 8 timeframes **and** median Sharpe gain
   ≥ +0.10.

## Secondary
Test results per coin and per year; 10 extra coins; delay, missed move, flips per true flip; gap between the choice and
the test-hindsight best; cross-timeframe comparison of the FILTER choices on common 1h returns; max drawdown.

## Amendment 1 (before any stage-27 number was computed)
**5m** bars (Binance 5m klines, same 20 coins) are added as a ninth timeframe on the user's request, with the same
grid, dev folds, selection rule and test. The primary endpoints A-C stay defined on the original eight timeframes;
5m is reported as secondary.

# Pre-registration — Stage 10: one model that tells the trend of ANY single stock
Written before any stage-10 number was computed.

## Goal
A single model, trained on stocks, that classifies the current trend (up / down) of any individual stock,
close to the ideal hindsight labeler, and that works on stocks AND years it has never seen.

## Data / label
`sp_prices.parquet`, 916 US stocks, daily OHLCV 1995-2026 (split-adjusted close).
Label = ideal hindsight trend (Viterbi oracle, k = 1, same as stages 3-4). Training rows only while the label
is final at the training cut-off + 24-bar embargo (no label leakage).
Features (all causal, leak-tested in earlier stages): the existing ~160 technical features (momentum,
Donchian, drawdown/rally, efficiency, EMA distance/slope, regression t-stat, SuperTrend, ADX/DI, RSI,
directional-change states, Bollinger, ATR%, volume, 4-day and 24-day higher-timeframe copies) PLUS market
context from the equal-weight index of all stocks (index momentum 21/63/252, index vs EMA200, breadth
= share of stocks above own SMA200) and relative strength (stock momentum - index momentum, 63/252).

## Ticker and time split (the anti-overfitting core)
- Test set A: the 50 large caps of stage 3 — never used in training or tuning.
- The remaining tickers are shuffled (seed 0) and split: 200 FIT, 100 VALIDATION, and every other one = test set B.
- Time: everything before 2011-01-01 is development; 2011-01-01 .. 2026-09 is the test period (once).
- Optuna (40 trials, TPE seed 0): LightGBM params + smoothing (span, hysteresis) trained on FIT tickers with
  data < 2006-01-01, scored on VALIDATION tickers 2006-2010, objective = median per-ticker MCC.
- Final model: frozen params, retrained on FIT + VALIDATION tickers with data < 2011-01-01. Never retrained.
- Baselines tuned by grid on the same validation data (median MCC): price vs EMA(n), EMA cross, SuperTrend,
  online directional-change; plus fixed EMA200 and the crypto-trained universal model C.

## Primary endpoint (both test sets, 2011-2026)
Model median per-ticker MCC > best tuned baseline's median MCC, AND model MCC > that baseline on a majority of
tickers with a two-sided sign test p < 0.05.

## Secondary (reported whatever they show)
Flip precision, flips per true flip, missed-move and give-back fractions, long/flat Sharpe vs buy & hold
per stock, distribution over tickers (share of tickers where the model wins), sub-periods 2011-2018 / 2019-2026.

## Amendment (user request, before any tuning or result): explicit choppiness features added
chop_14, chop_50 (Choppiness Index), ema20_cross_rate_50 (share of the last 50 days with a close crossing EMA20),
var_ratio_5_250 (Lo-MacKinlay variance ratio, >1 trending, <1 choppy), trend_r2_50 / trend_r2_150
(R^2 of log price on time). All rolling windows end at t (causal).

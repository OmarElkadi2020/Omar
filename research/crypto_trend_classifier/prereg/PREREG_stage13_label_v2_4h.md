# Pre-registration — Stage 13: label v2 (user-approved) on the 4h timeframe
Written before any stage-13 model was trained or scored.

## Label (approved by the user, tc/label_v2.py, unchanged)
score in [-1,1] = major trend (oracle k=1) x pivot emphasis (1 at the reversal candle, decays to 0.35 with
tau = 15% of the trend length) x forward volume-flow confirmation (next 10 bars; 0.5..1.0) + counter-moves
(minor oracle k=0.3 swings against the major trend, size relative to a trend-flipping move; rebounds inside
a DOWN trend count half), clipped. Training rows only while both oracles are final at the cut-off, 10 bars
for the forward flow, then a 24-bar embargo. No price inversion augmentation (the label is asymmetric).

## Data (all 4h)
Crypto: BTC ETH BNB XRP ADA TRX DOGE ZEC BCH SOL (Binance) + BTC Bitstamp before 2017-08.
Oanda 4h (tick volume): 6 FX pairs, 8 indices + DAX (histdata), gold. All end 2020-05.
- Held-out coins (never trained on, any date): ADA, XRP, DOGE.
- Development: everything before 2021-01-01. Test: 2021-01-01 .. 2026-09-23 (once).

## Features (causal, truncation leak test before use)
The 161 frozen model-C columns + stage-12 features (vol-normalised returns, normalised MACD, CUSUM, BOCPD,
OBV/AD) + choppiness (stage 10) + NEW liquidity-location features: share of the last N bars' volume traded
below the current close (N = 30, 120), (close - VWAP_N)/ATR14, (close - volume point-of-control_120)/ATR14.

## Model and tuning (development data only)
LightGBM regression on the score. Optuna 30 trials (TPE seed 0): train on training instruments < 2019-01-01,
validate on training coins 2019-01-01 .. 2020-12-31, objective = Spearman correlation of prediction vs label.
Signal conversion (stage-12 H and C grids) chosen on the same validation window by the primary metric under
the budget. Final model retrained on all training instruments < 2021-01-01.

## Baselines (parameters chosen on the same validation window, same budget)
EMA cross, price vs EMA, SuperTrend, online directional-change, pure CUSUM (stage-12 grids);
crypto model C walk-forward states (fixed); stage-12 stock model applied unchanged (fixed conversion).

## Primary metric / endpoint
Truth = binary oracle k=1. Metric = missed-move fraction (stage 12), budget = median flips per true flip <= 2.0.
Cells = coin x calendar year, 2021-2026. PRIMARY: on (i) held-out coins and (ii) training coins, the model's
median missed-move is lower than the best baseline's AND lower in a majority of cells (two-sided sign test
p < 0.05), with test flips-per-true-flip <= 2.5.
Secondary: delay (bars), give-back, MCC, Spearman vs label v2, long/flat Sharpe vs buy & hold (0.1% cost).

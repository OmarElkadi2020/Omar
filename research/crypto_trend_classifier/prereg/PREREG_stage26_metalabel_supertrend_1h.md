# Stage 26 pre-registration: meta-labeling SuperTrend as a 1h long trend filter

Written and committed before any 1h data are processed or any stage-26 number is computed.

## Question
SuperTrend decides the side; a secondary model (meta-labeling, López de Prado AFML ch. 3; Joubert, JFDS 2022)
decides whether to **accept each SuperTrend up-flip**. Using 1h and 4h information SuperTrend cannot see (order flow,
derivatives positioning, cross-market, multi-timeframe price state), does the filtered 1h trend state classify the
trend **more accurately than plain SuperTrend**, with fewer false signals and a better long filter?

## Assets, bars, periods
* Evaluation: BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX (USDT spot). Extra training-only coins: DOT LTC BCH ATOM
  NEAR UNI FIL ETC XLM AAVE. Decision bars: **1h** (Binance spot 1h klines).
* Dev: folds 2022 and 2023 (each trained strictly before). Test: **2024-01-01 … 2026-08-31**, annual walk-forward
  (refits at 2024-01-01, 2025-01-01, 2026-01-01). Test run once.

## Truth
Oracle trend state on 1h closes, `labels.oracle_labels(k = 1)` (project standard B1). The last 240 bars of the test
window are not evaluated (labels not final). Secondary: k = 2 (slower trend definition).

## Primary signal and events
Primary = SuperTrend(period, multiplier) on 1h. An **event** is an up-flip bar (state 0 → 1). The up-segment runs
from the flip bar to the next down-flip, capped at 720 bars (30 days).
Meta-label candidates (one chosen on dev):
* `ret`: segment return (close at flip → close at segment end) − 20 bp > 0 (the trade was worth taking);
* `oracle`: more than half of the segment's bars are oracle-up.
Training uses only events whose segment ended ≥ 24 bars before the refit date; for `oracle`, labels are computed on
data truncated at the refit date and only bars final at that date (minus 24) are used.

## Filtered state (META)
At each up-flip the model gives p. If p ≥ τ the segment is ON (up) until SuperTrend flips down; otherwise the
whole segment stays OFF. Down-flips are never vetoed (long filter).

## Features at the event bar (all known at the bar close; truncation leak test must return 0)
1. 1h price/volume set (stage-12/14 builder at 1h with 4h and 1-day aggregates, Bollinger/ATR, choppiness,
   market context), 1h order flow (taker-buy share 6/24/72 h, trade-count z, perp taker share, perp/spot volume).
2. The complete stage-25 4h frame (price, flow, derivatives, cross-market), taken from the **last completed 4h bar**.
3. Event context: ATR / price, close − SuperTrend line (in ATR), bars since previous up-flip, previous
   up-segment return and length, flips in last 168 bars, share of up bars in last 168 bars, 4h SuperTrend(48,4) state.

## Model and tuning (dev only)
LightGBM binary, pooled over 20 coins. Optuna 40 trials (TPE, seed 0) over: primary ∈ SuperTrend period {10, 14, 24,
48} × multiplier {2, 3, 4}; label ∈ {ret, oracle}; LightGBM leaves 4–31, lr 0.01–0.1, trees 100–500, min leaf
20–500, feature/bagging fraction 0.3–1, L2 1e-3–100; τ ∈ {0.20, 0.25, …, 0.70} chosen per trial.
Dev objective for everyone (stage-25): mean MCC over the 10 evaluation coins − 0.1·max(0, median flips-per-true-flip − 2).

## Benchmarks
BEST = best plain 1h SuperTrend on the same dev objective over period {7, 10, 14, 24, 48} × multiplier
{1.5, 2, 3, 4, 5}. Also reported: SuperTrend(10,3) 1h, META's own primary unfiltered, and the stage-25 choice
SuperTrend(48,4) on 4h projected onto 1h bars (last completed 4h bar).

## Primary endpoint (test; all three must hold)
1. **Accuracy:** median MCC(META) − median MCC(BEST) ≥ **+0.02**, and META > BEST in ≥ 7 of 10 coins.
2. **Signal precision:** pooled share of ON segments with net return > 0 (20 bp round trip) of META ≥ BEST's
   **+ 5 percentage points**.
3. **Filter efficiency:** equal-weight long-only portfolio (position = state, 1-bar lag, 10 bp per change) Sharpe of
   META ≥ BEST's.

## Secondary
Delay, missed move, flips per true flip, give-back; per coin, per year; truth k = 2; ablation without the 4h
derivatives/flow/cross blocks; META vs its own unfiltered primary (the pure meta-labeling effect).

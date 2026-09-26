# Stage 25 pre-registration: every available feature, beat the trend indicators on accuracy AND speed

Written and committed before any stage-25 data are processed or any number is computed.

## Question
With **new information the indicators cannot see** (derivatives positioning and order flow) added to all price
features, can a model classify the trend of large-cap crypto noticeably **more accurately and faster** than the best
classic indicator, without more false signals?

## Assets, bars, periods
* Evaluation: BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX (USDT). Extra training-only coins: DOT LTC BCH ATOM NEAR UNI
  FIL ETC XLM AAVE. Bars: **4h** (decision every 4h); daily/weekly context enters as features (MTF).
* Dev: folds 2022 and 2023 (each trained strictly before). Test: **2024-01-01 … 2026-08-31**, annual walk-forward.
  (Crypto 2022-26 was used by earlier stages for other questions; the bar below is set high for that reason.)

## Truth
Oracle trend state on 4h closes (`labels.oracle_labels`, k = 1, switching cost k·σ·√24 — the project's standard B1
label). Evaluation ignores the last 60 bars of the test window (labels not final). Training rows respect finality
with a 24-bar embargo before each refit date.

## Features
1. Price/volume (all timeframes): the stage-12 set (208 columns incl. 4h, 16h, 4-day aggregates, CUSUM, BOCPD,
   choppiness, MACD-norm, OBV/AD) plus daily-bar trend block (EMA50/200 distance, TSMOM 30/90, Donchian position).
2. **Order flow (new):** spot & perp taker-buy share (6/24/72-bar), number-of-trades z-score, perp/spot volume ratio.
3. **Derivatives positioning (new):** funding rate (last, 3-day sum, 30-day z), open interest log-change
   (1/3/7 days) and OI / 30-day quote volume, top-trader and global long/short ratios (level, 1-day change),
   taker long/short volume ratio, perp-spot basis (premium index: last, 1-day mean, 30-day z).
4. **Cross-market (new):** BTC's own derivatives block for every coin, ETH/BTC trend, top-100 breadth
   (share above daily EMA50) and dispersion from the stage-20 panel (known after each daily close).
All derivatives data are aligned to the 4h bar close at which they are published (5-min metrics → last value
≤ bar close; funding at settlement). A truncation leak test must return 0 before training.

## Model and conversion
LightGBM binary (up-state probability), pooled over the 20 coins; Optuna 30 trials on dev.
State = hysteresis on EWM(p): ON (up) above θ_in, OFF below θ_out; span, θ tuned on dev.

## Benchmarks (all parameters tuned on the same dev folds, same objective)
EMA price cross (n ∈ 20, 50, 100, 200, 400), EMA pair cross (fast 5/10/20/50 × slow 50/100/200/400), SuperTrend
(period 7/10/14/24/48 × factor 1.5/2/3/4/5), Donchian breakout state (20/55/100), TSMOM (n ∈ 12, 42, 84, 180 bars),
MACD(12,26,9)>signal, CUSUM on returns (κ 0.1/0.25/0.5/1 × h 2/4/8/16/32).
Dev objective for everyone: mean MCC over the 10 evaluation coins, penalised when median flips per true flip > 2
(MCC − 0.1·max(0, FPT − 2)). The single best indicator on dev = "BEST".

## Primary endpoint (test 2024-01 … 2026-08; all three must hold)
1. **Accuracy:** median MCC(model) − median MCC(BEST) ≥ **+0.05**, and model MCC > BEST in ≥ 8 of 10 coins.
2. **Speed:** median detection delay (bars) of the model ≤ 0.8 × BEST's (≥ 20 % faster).
3. **False signals:** median flips per true flip of the model ≤ 1.1 × BEST's.

## Secondary
Missed-move fraction, give-back; the same comparison against every indicator family's best; feature-block ablation
(price only / + order flow / + derivatives / + cross-market) on test; per-coin table; frontier plot (MCC vs delay).

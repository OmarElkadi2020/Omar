# Stage 15 pre-registration: forward-return trend model, alpha test on stocks + crypto

Written and committed **before** any stage-15 model is trained or any test-period return is computed.

## Why a new target
Stages 10–14 trained on *state* labels (oracle trend segments). Classifying the current state well
(MCC wins vs EMA baselines) did not turn into returns above buy & hold, because the state label is,
by construction, known late. Here the model predicts the thing that pays: the **forward
volatility-normalised return** over the next H bars.

## Data (all already on disk; no new download)
* Stocks: every ticker in `cache_stage12` (≈861 US large/mid caps, daily, 1995–2026-09).
  Features = the 208 frozen stage-12 columns (`FROZEN_stage12.json['cols']`), which already contain
  multi-timeframe blocks (`h4_*`, `h24_*` = 4-bar and 24-bar aggregated bars) and the equal-weight
  market context. PnL / targets use `adj_close` (dividends + splits); daily log returns clipped ±0.4.
* Crypto: the 10 Binance coins used before (BTC ETH BNB XRP ADA TRX DOGE ZEC BCH SOL), built on
  **two timeframes, 4h and 1D**, with the same 208 columns (`stage14.frame`, coin-basket context).
* MTF training: one pooled model per retrain date, rows from stocks-1D, crypto-1D and crypto-4h.
  Horizons and features are in bars, so the same model reads every timeframe.

## Target
`y_t = clip( log(P_{t+H}/P_t) / (sigma_t * sqrt(H)), -4, 4 )`, sigma_t = EWM std (span 60) of past
1-bar log returns. LightGBM regression (L2). H is chosen on dev data from {10, 20, 40}.

## Signal → position
`pos_t = clip( EWM_span(pred)_t / s, -1, 1 )` (long/short), s = std of the model's in-sample
predictions (training rows only). Position set at bar-t close, earns bar t+1 return.
Costs per unit of turnover: stocks 5 bp, crypto 10 bp (benchmarks pay the same).

## Benchmarks (the alpha must survive all of them at once)
Same per-asset, same costs, equal-weighted across assets:
1. BH: buy & hold (pos = 1)
2. TSMOM: sign(252-bar return) (Moskowitz–Ooi–Pedersen)
3. EMA50: sign(close − EMA50)
4. BAZ: mean of φ(x)=x·exp(−x²/4)/0.89 over the normalised MACDs 8/24, 16/48, 32/96 (Baz et al. 2015)

## Development (tuning) — data before 2011-01-01 only, stocks only
* Train on the stage-10 `fit` tickers (200), rows whose forward window ends before 2006-01-01.
* Validate on the `val` tickers (100) over 2006-01-01 … 2010-12-31.
* Objective = Newey–West t-stat of α in the spanning regression below, on the val EW portfolio.
* Optuna TPE, 40 trials, seed 0: H, smoothing span {1,3,5,10}, num_leaves, learning rate, trees,
  min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l2. Frozen to `FROZEN_stage15.json`.

## Test — expanding-window walk-forward, hyper-parameters frozen
* One model per calendar year Y = 2011 … 2026, trained on all rows (all stocks, all coins, both
  crypto timeframes) whose forward window ends at least 5 bars before Y-01-01. Predicts year Y only.
* Stock rows are sub-sampled every 5th bar (overlapping targets); crypto 4h every 5th bar; crypto 1D
  every bar. Sample weights: crypto-4h and crypto-1D rows each carry 12.5% of the total weight when
  present, stocks the rest.
* Stocks test: 2011-01-01 … 2026-09-22 (all tickers). Crypto test: 2021-01-01 … 2026-09-23.

## Spanning regression (the alpha definition)
`r_model,t = α + β1·r_BH,t + β2·r_TSMOM,t + β3·r_EMA50,t + β4·r_BAZ,t + ε_t`
on the equal-weighted portfolio returns, Newey–West HAC (lag = 20 days; 120 bars on 4h).

## Primary endpoints (goal met only if both hold)
* **P1 stocks (1D):** α > 0 with NW t ≥ 3.0 (Harvey–Liu–Zhu hurdle, because many stages came before).
* **P2 crypto:** α > 0 with NW t ≥ 2.0 on **both** 4h and 1D (conjunction, no extra tests).

## Secondary (reported, not used for the verdict)
Plain Jensen α vs BH only; per-asset α sign counts (sign test); Sharpe of model vs benchmarks;
long/flat variant; per-year α; daily cross-sectional rank IC for stocks; stage-15 model applied to
the 50 `A` large caps only.

## Known caveats written in advance
Stock universe = today's constituents (survivorship). It inflates buy & hold, not the α of a
long/short timing overlay relative to buy & hold, but the verdict is stated with it.

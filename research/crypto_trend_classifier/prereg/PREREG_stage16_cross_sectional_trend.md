# Stage 16 pre-registration: cross-sectional trend model (which assets trend up *more*), crypto + stocks

Written and committed before any stage-16 model is fitted or any stage-16 return is computed.

## Motivation (literature, not our data)
* Gu, Kelly & Xiu (RFS 2020): ML gains in returns come from the **cross-section** (relative returns),
  trees capture non-linear interactions of momentum / volatility / liquidity signals.
* Liu, Tsyvinski & Wu (JF 2022): crypto cross-section is spanned by market, size and momentum;
  "A trend factor for the cross section of cryptocurrency returns" (JFQA 2024) combines MAs of many
  horizons. So our alpha must be measured *after* those known factors.
* Stage 15 (same session) showed that single-asset timing alpha beyond TSMOM/EMA/MACD is ~0 on dev.

## Data
* Crypto: every Binance spot `*USDT` pair in the public archive (data.binance.vision), **including
  delisted pairs** (no survivorship), 4h klines 2017-08 … 2026-08. Excluded a priori: leveraged tokens
  (UP/DOWN/BULL/BEAR), fiat/stable coins, wrapped duplicates (WBTC, WBETH, BETH).
* Stocks: the 861 tickers of `cache_stage12` / `data/sp_prices.parquet` (current constituents →
  survivorship caveat), daily OHLCV + adj_close 1995 … 2026-09.

## Universe (per date, causal)
* Crypto: ≥ 60 daily bars of history and top 100 by 30-day median dollar volume.
* Stocks: ≥ 252 bars of history, all eligible tickers.

## Features (computed per asset, then ranked cross-sectionally per date to [-0.5, 0.5])
Daily bars: log returns over 1, 3, 7, 14, 30, 60, 90, 180 (and 252 stocks) bars, vol-normalised;
volatility 7/30/90 and ratio; MAX (largest 1-bar return, 30); skew 30; EMA distances 10/30/100 in σ
units; drawdown from / rally off 90-bar extremes; Donchian position 20/60; efficiency ratio 14/60;
choppiness 14; trend R² 30/90; RSI 14; log dollar volume (size), dollar-volume change 7/30; beta and
residual momentum vs the equal-weight universe (60 bars); age.
**Second timeframe (MTF):** crypto 4h bars (RSI 14, ER 42, 12-bar vol-normalised return, chop 42,
position in 42-bar range); stocks weekly bars (4- and 13-week return, weekly RSI 14, weekly ER 13).
Market context kept raw (not ranked): EW-universe returns 7/30/90, BTC (crypto) or EW (stocks) 30-bar
return, breadth (share with positive 30-bar return), cross-sectional dispersion.

## Target
Cross-sectional percentile rank (−0.5…0.5) of the forward H-bar return, measured from the
execution price. H ∈ {3, 7, 14, 30} chosen on dev. LightGBM regression.

## Execution realism
* Crypto: features at the 00:00 UTC daily close; **trade at the close of the next 4h bar (04:00)**,
  i.e. all returns are measured 04:00→04:00. Costs 10 bp per unit of one-way turnover.
* Stocks: features at close t, **trade at the next open**; returns open→open (split/dividend adjusted
  with adj_close/close). Costs 5 bp per unit of one-way turnover.

## Portfolio
Each day: long the top quintile, short the bottom quintile of predictions, equal-weight, gross 1+1.
The traded book is the average of the last `hold` daily books (`hold` ∈ {1, 3, 7}, dev-tuned) to
limit turnover. A long-only book (top quintile, gross 1) is reported as well.

## Development (tuning) and test
* Crypto: train on rows whose target ends before 2021-01-01 (−7 days embargo), validate 2021;
  test **2022-01-01 … 2026-08-31**.
* Stocks: train < 2006-01-01 (−7 days), validate 2006–2010; test **2011-01-01 … 2026-09-22**.
* Optuna TPE, 30 trials per market, seed 0, objective = NW t-stat of the long-short α on validation.
* Test = expanding walk-forward, one refit per calendar year, frozen hyper-parameters.

## Alpha definition (spanning regression, NW HAC lag 20)
Long-short daily net returns on factors built on the same universe, same execution timing, gross:
* Crypto: EW market, BTC, SIZE (small − big dollar-volume tercile), CMOM (3-week momentum tercile
  long-short, Liu–Tsyvinski–Wu), STREV (1-week reversal tercile long-short).
* Stocks: EW market, SIZE, UMD (12-1 month momentum), STREV (1-month reversal), LOWVOL (low − high
  60-day vol).

## Primary endpoints (goal met only if both hold)
* **P1 crypto:** long-short α > 0, NW t ≥ 3.0, 2022-01 … 2026-08.
* **P2 stocks:** long-short α > 0, NW t ≥ 3.0, 2011-01 … 2026-09.

## Secondary
Net Sharpe, CAGR, max DD of long-short and long-only; long-only α vs the same factors; per-year α;
one-extra-day delay robustness; rank IC; stocks top-50 subset; turnover.

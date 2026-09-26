# Stage 20 pre-registration: long-only trend classifier that must beat every long-only indicator

Written and committed before any stage-20 feature, model or return is computed.

## What changes vs stage 16
Stage 16 predicted the *relative* rank → great long-short, weak long-only (it picks the best of a falling
alt market). Stage 20 predicts the **absolute up-trend** of each asset and may hold **cash**:
* Target: `y = 1{forward H-day return > 0}` measured from the execution price, sample weight = |forward
  return| (big trends matter more, tiny moves little). LightGBM binary classifier → p(up-trend).
* Features: the stage-16 cross-sectional ranks **plus the raw (scale-free) time-series values** of the same
  trend/vol/chop features, 4h MTF block, and market context (EW market, BTC, breadth) — so the model can see
  both "better than the others" and "actually trending up". (beta bug fixed.)
* Book: each day hold, equal weight, the top-K assets by p among those with p > τ; empty slots = cash (0 %).
  Averaged over `hold` days. H, K, τ, hold and LightGBM params are dev-tuned.

## Market and periods (crypto)
Binance USDT spot, incl. delisted, top-100 by liquidity, execution at the next 4h close, 10 bp costs.
* Dev: expanding folds, validate 2019, 2020 and 2021 separately, each trained on data before it
  (7-day + horizon purge). Optuna TPE 40 trials seed 0. Objective = mean over the 3 folds of the α t-stat
  below.
* Test: annual walk-forward refits, **2022-01-01 … 2026-08-31**.
* Honest note: this crypto window was already used in stages 16 and 19. Hurdle raised to t ≥ 3.0.

## Long-only indicator benchmarks (all on the same universe, same execution, same costs, cash when out)
Per coin, "in" when the indicator is bullish; the book holds all "in" coins equally (or cash):
EMA50 (close > EMA50), EMA200, golden cross (SMA50 > SMA200), SuperTrend(10, 3), Donchian 20/10
(enter on a 20-day high, exit on a 10-day low), MACD(12,26,9) > signal, TSMOM-30 (30-day return > 0),
TSMOM-90, ADX(14) > 25 with +DI > −DI. Cross-sectional: top-20 % by 30-day return (always invested) and
dual momentum (top-20 % by 30-day return **and** 30-day return > 0, else cash). Passive: BTC buy & hold,
BTC with EMA50 filter, EW universe buy & hold.

## Primary endpoint (both must hold)
1. α > 0 with NW (lag 20) t ≥ 3.0 of the model's daily net returns regressed on **all** the benchmark
   strategies above at once, 2022-01 … 2026-08.
2. The model's Sharpe is higher than the Sharpe of **every** benchmark strategy.

## Secondary: "does it classify the trend correctly?"
For every strategy: precision of its buy signals (share of held coin-days whose next-H return > 0),
average forward H-day return of held coin-days, share of "false buys" (held coin-days with next-H return
< −5 %), time in market, turnover, max DD, per-year returns.

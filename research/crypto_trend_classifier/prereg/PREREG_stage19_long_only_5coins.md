# Stage 19 pre-registration: long-only rotation among BTC, ETH, SOL, LINK, BNB (user-chosen coins)

Written before any stage-19 number is computed. No training, no tuning: the frozen stage-16 crypto model's
walk-forward scores (already out-of-sample, 2022-01 … 2026-08) are reused as they are.

* Universe: BTCUSDT, ETHUSDT, SOLUSDT, LINKUSDT, BNBUSDT (chosen by the user).
* Signal: each day rank the 5 coins by the stage-16 score. Execution 04:00 UTC (next 4h close), 10 bp costs,
  book averaged over the frozen `hold` = 3 days, simple-return accounting.
* **Primary: hold the top-2 of 5, equal weight, always invested.**
  Benchmarks: equal-weight 5 (daily rebalanced) and BTC buy & hold.
  α = intercept of daily net returns on [EW5, BTC], Newey–West lag 20. Primary endpoint: α > 0, t ≥ 2.0.
* Secondary (reported, not used for the verdict): top-1, top-3; "cash filter" variant = hold a top-2 coin only
  if its rank in the full top-100 universe is in the top 20 %, otherwise that slot is cash; per-year returns.

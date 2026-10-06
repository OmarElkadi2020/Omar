# Stage 31 pre-registration: forward test on data that did not exist when the rules were frozen

Written and committed on 2026-10-06, before any data after 2026-09-30 is downloaded or used.

## Why
Stage 30 (test 2024-01 … 2026-09) gave the largest money gap of the project but p = 0.09. The exploratory stage 30b
on the same window showed POLICY's gain over SuperTrend is about the size of plain volatility sizing:
SuperTrend + vol sizing had Sharpe 0.82 vs POLICY 0.77; POLICY's direction with vol sizing had 0.90. That window is
now used up. Only data from after the freeze can judge these rules.

## Frozen rules (no change, no tuning, ever, in this stage)
* **A: SuperTrend(48, 5) on 4h, long/flat, × stage-27 vol sizing** (2 % daily-vol target from a trailing 30-day std
  of 4h returns, cap 2×). The strongest simple rule.
* **B: POLICY** (stage-30 long/flat model, `FROZEN_stage30.json`), continuous position. The 2026 model is the one
  saved in `models/stage30/policy_lf_2026.npz` (trained on data before 2026-01-01, 5 seeds). On 1 Jan of each later
  year it is refit with the frozen stage-30 procedure on all data before that date, and saved.
* **C: POLICY direction (position ≥ 0.5) × the same vol sizing as A.**
Also shown, not tested: SuperTrend unsized, buy & hold.

## Setting
10 evaluation coins (BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX), equal weight, 4h bars from Binance spot 1h data
(`python -m tc.stage31 update`, monthly files then daily files), cost 0.1 % per side on |Δposition|.
Forward window starts **2026-10-01 00:00 UTC**. Code: `tc/stage31.py`.

## Endpoint
**Primary, evaluated once at 2027-10-01 (12 months):** Sharpe(C) − Sharpe(A) > 0 with paired stationary block
bootstrap one-sided p < 0.05. Secondary: B − A; per-coin counts; max drawdown; CAGR.
Earlier runs are progress reports only: they are recorded, but no decision is taken from them and nothing is changed.
A second look at 2028-10-01 (24 months) is pre-registered as a replication on the full 24 months.

## Power, said in advance
One year of 10 correlated coins can only detect a Sharpe gap of roughly 1. A real gain of 0.1-0.3 will most likely
read "not met" at 12 months. The value of this stage is that the result cannot be shaped by any choice we make.

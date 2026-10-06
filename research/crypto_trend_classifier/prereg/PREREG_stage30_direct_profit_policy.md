# Stage 30 pre-registration: a model trained directly on profit after costs (no label)

Written and committed before any stage-30 model or result is computed.

## Question
Stages 1-29 trained classifiers on a trend label and converted probabilities into positions. Here the model outputs
the **position itself** and is trained to maximise the Sharpe ratio of its own P&L **after transaction costs**
(the "deep momentum network" set-up: Lim, Zohren & Roberts, JFDS 2019). No label, no threshold.
Does this make more money than SuperTrend(48, 5), and more than the label-trained model of stage 29?

## Data and features (identical to stage 29)
4h OHLC of BTC ETH BNB XRP SOL from `trend_compass.html`, 2019-01-01 … 2026-09-23; stage-29 price-only features
(leak test 0 mismatches). Features are standardised with the mean and std of the training rows only; missing → 0.

## Model (`tc/stage30.py`, plain numpy, no new dependency)
MLP: features → hidden layer (tanh) → one output z.
* long/flat: position = sigmoid(z) ∈ [0, 1];  long/short: position = tanh(z) ∈ [−1, 1].
* P&L of bar t (per coin, in time order): pos[t]·r[t+1] − c_train·|pos[t] − pos[t−1]|, r = simple 4h return.
* Loss = −Sharpe of all coin-bars pooled + L2 · ||weights||². Exact analytic gradient (the cost term couples
  neighbouring bars; its gradient is computed along each coin's sequence). Full-batch Adam, lr 0.003.
* Epochs: early stopping on the last 20 % (in time) of the training period, patience 30, max 400.
* Ensemble of 5 seeds; the position is the mean of the 5 positions.
* Training rows: every bar whose next return closes before the cut-off. No label, so no finality embargo.

## Dev tuning (folds 2022 and 2023, each trained only on earlier data)
Grid: hidden {8, 32} × L2 {1e-4, 1e-3, 1e-2} × training cost c_train {0.1 %, 0.3 %} (a higher training cost is a
way to make the model trade less). Chosen separately for long/flat and long/short by the mean dev portfolio Sharpe
(evaluated at the real 0.1 % cost). Frozen in `FROZEN_stage30.json`.

## Test (run once)
2024-01-01 … 2026-09-23, refits on 1 Jan 2024, 2025, 2026. Evaluation = the stage-29 money evaluation
(`tc/stage29.py` functions): equal-weight portfolio, cost 0.1 % per side on |Δposition| (0.2 % stress), CAGR,
Sharpe, max DD, Calmar, turnover, capture of the hindsight label's P&L, regret split, money-weighted accuracy,
paired stationary block bootstrap (block 20 days, 10,000) for Sharpe differences, Newey-West alpha vs buy & hold.
Positions are continuous (the model sizes its bets). A discretised copy (long/flat: pos ≥ 0.5 → 1; long/short:
sign) is reported for trade statistics.

## Primary endpoint
In **either** mode (long/flat or long/short, Bonferroni: p < 0.025 each):
1. Sharpe(POLICY) − Sharpe(SuperTrend(48, 5)) > 0 with bootstrap one-sided p < 0.025, and
2. POLICY has the higher Sharpe in ≥ 4 of 5 coins in that mode.

## Secondary
POLICY vs stage-29 EQUAL and WEIGHTED-24h (frozen stage-29 settings, re-run here); vs buy & hold; 0.2 % cost;
discretised POLICY; per year; per coin; turnover.

## Power, said in advance
Same as stage 29: only a Sharpe gap of about 0.6 can reach significance on 2.7 years of 5 correlated coins.

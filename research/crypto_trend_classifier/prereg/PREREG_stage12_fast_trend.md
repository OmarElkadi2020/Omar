# Pre-registration — Stage 12: faster trend detection (labels x features x signal conversion)
Written before any stage-12 feature, label or result was computed. Plan: PLAN_stage12_questions.md.

## Truth and primary metric
Truth = ideal hindsight binary trend, Viterbi oracle k = 1 (as in stages 3-11), final labels only.
PRIMARY metric = median over tickers of the **missed-move fraction**: for every true trend segment (>= 5 bars),
the share of its log move that happens before the indicator first agrees with it (0 = caught at the turn,
1 = never caught). Evaluated at a fixed **false-signal budget**: median flips per true flip <= 2.0.
Secondary: detection delay in bars, exit give-back, MCC, flip precision, long/flat Sharpe vs buy & hold.
Neutral (0) never "agrees"; every state change counts as a flip (conservative for 3-state outputs).

## New features (causal, leak-tested by truncation before use)
Vol-normalised returns r_k/(sigma*sqrt(k)), k = 1,5,21,63,126,252; normalised MACD (Baz) 8/24, 16/48, 32/96;
two-sided CUSUM of standardised returns (kappa 0.25/0.5/1.0: S+, S-, bars since last alarm h=4);
Bayesian online changepoint (Adams-MacKay, Gaussian mean, hazard 1/100, run length <= 300):
P(run length < 5), P(< 20), expected run length; OBV and Accumulation/Distribution 20/60-bar normalised slopes.

## Candidate labels (same features, same frozen stage-10 LightGBM params, no re-tuning)
B0.5, B1, B2 = oracle binary k 0.5/1/2 | B1w = B1 with weight 3 on the first 20% of each segment |
T1 = oracle ternary (up/neutral/down, up<->down must pass neutral, neutral earns 0, switch cost k=1) |
TS = trend-scanning sign (forward windows 5..60 bars), sample weight |t| | RV = remaining value regression
(log move left to the end of the k=1 segment / (sigma_21)). Score in [-1,1]: 2p-1, p_up-p_down, tanh(pred/sd).

## Signal conversion (grids, chosen on validation only)
H: EWM(span) + 3-state hysteresis (enter at +-theta_in, leave at theta_out), span {1,3,5,10},
theta_in {0.1..0.5}, theta_out {-theta_in (binary), 0, theta_in/2}.
C: CUSUM on the score, kappa {0,0.1,0.2,0.3}, h {0.5,1,2,4,8}.
(Meta-labeling deferred to a later stage.)

## Baselines (same budget, params chosen on the same validation data)
EMA cross, price vs EMA, SuperTrend, online directional-change (stage-10 grids), pure CUSUM on standardised
returns (kappa {0.1,0.25,0.5,1}, h {2,4,8,16,32}), and the stage-10 calm model (fixed).

## Selection: CPCV on development data only (< 2011)
Time 1996-2010 in 6 blocks, all 15 pairs of test blocks; train on FIT tickers in the other 4 blocks (21-bar
embargo around test blocks), score on VALIDATION tickers inside the 2 test blocks. For each candidate
(label x conversion x params): mean over the 15 splits of the primary metric, subject to the budget.
The best (label, conversion, params) is frozen, retrained on FIT+VAL tickers < 2011.

## Test (once)
A = 50 large caps, B = 510 other unseen stocks, 2011-2026; top-5 crypto (BTC ETH BNB XRP SOL) 1D 2019-2026
and 4h 2019-2026 with the stage-11 conventions.
PRIMARY ENDPOINT: on A and on B, the selected indicator's median missed-move at its frozen setting is lower
than the best baseline's (chosen the same way), with a two-sided per-ticker sign test p < 0.05, while its
test flips-per-true-flip stays <= 2.5.

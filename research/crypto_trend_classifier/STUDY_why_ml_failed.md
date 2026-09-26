# Why did ML fail to beat SuperTrend? (diagnostic study on stage-26 data, exploratory)
Code: `tc/diag26.py`; results: `results_diag26_*.csv`. 10 majors, 1h, test 2024-01 … 2026-08 unless noted.

## 1. Where SuperTrend(48,4)'s errors come from, and the ceiling for any filter
| state | MCC | wrong bars | of which lag |
|---|---|---|---|
| SuperTrend(48,4) 1h | 0.371 | 31 % | 49 % |
| PERFECT meta-label (knows the future, oracle label) | 0.560 | 23 % | 57 % |
| PERFECT meta-label (knows the future, "trade was profitable" label) | 0.446 | 31 % | 77 % |
| true trend delayed by 18 bars (SuperTrend's delay, no whipsaws) | 0.755 | 12 % | 98 % |
Half the error is unavoidable lag, half whipsaw. A filter can only remove up-flips, so even a clairvoyant one tops out
at 0.56; and "most accurate" and "most profitable" are different targets (0.56 vs 0.45).

## 2. How good a meta-model must be (noisy clairvoyant labeler, best threshold chosen with hindsight)
| AUC | 0.55 | 0.59 | 0.65 | 0.68 | 0.77 | 0.82 | 0.92 |
|---|---|---|---|---|---|---|---|
| MCC gain vs SuperTrend | −0.028 | −0.019 | +0.008 | +0.001 | +0.025 | +0.046 | +0.089 |
Break-even needs AUC ≈ 0.65; a useful +0.03 needs ≈ 0.78.

## 3. How good our meta-model actually is (out-of-sample AUC, 2022-2026 average)
All 475 features 0.534 (profit label) / 0.536 (oracle label); shuffled-label null 0.503 / 0.509.
By block: 1h price 0.537 / 0.542, 4h price 0.537 / 0.524, event context 0.511 / 0.534, 4h flow 0.510 / 0.522,
1h flow 0.506 / 0.505, **4h derivatives 0.502 / 0.494, cross-market 0.493 / 0.515 (= null)**.
→ ~0.54 predicts a loss of ≈ −0.02 MCC; the test gave −0.019. The failure is fully explained by the weak signal.

## 4. What is predictable at all (LightGBM on all features, 1h bars, walk-forward 2024/25/26)
| target | AUC |
|---|---|
| direction of the next 24 h | 0.525 / 0.527 / 0.535 |
| size of the next 24 h move (vs its volatility) | 0.590 / 0.580 / 0.577 |
| current trend state (the classification task itself) | 0.812 / 0.767 / 0.757 |
The state is "predictable" only through what has already happened (top features: Donchian position, SuperTrend(24,4),
EMA distance: the model rebuilds indicators). The future direction, which is what fixes lag and whipsaws, is not.

## 5. Relations are weak and unstable
Best single feature |AUC − 0.5| ≈ 0.05-0.06; median 0.015 (≈ noise for ~3,000 events). For the profit label the
rank correlation of feature effects between 2022-23 and 2024-26 is **−0.16**; the top-20 features of 2019-21 keep
their sign in 2024-26 only 55 % of the time (coin flip). For the oracle label it is 0.31 / 80 %, but those features
are momentum measures SuperTrend already contains, and their effect decays (e.g. EMA-20 distance 0.070 → 0.001).

## 6. The payoff is in the tail, and the evidence is thinner than it looks
1,852 test trades: win rate 38 %, mean win +6.0 %, mean loss −3.2 %. **The top 10 % of trades make 484 % of the total
P&L; the other 90 % lose.** META rejected 133 trades (7 %) incl. 5 % of the big winners; Spearman(p, return) = 0.06.
Mean pairwise hourly correlation of the 10 coins 0.66 → **effective independent coins ≈ 1.4**.

## 7. What does work (exploratory): size, not side
Inverse-volatility sizing of the same SuperTrend(48,4) (no ML): Sharpe 0.60 → 0.87, max DD −47 % → −30 %,
volatility halved, same CAGR. Also stage 16: cross-sectional ranking (common market direction removed) t = 5.3.

## Conclusion
Trend classifiers err mainly by lag and whipsaw; fixing either needs a forecast of future direction, and at 1h-24h
horizons direction is close to unpredictable with public price, flow, derivatives and cross-market data
(AUC ≈ 0.53, unstable). A filter needs AUC ≈ 0.65 just to break even and ≈ 0.78 to be useful. What IS predictable
is the size of moves (volatility) and relative performance across coins, and those are where edges were found.

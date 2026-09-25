# Pre-registration — Stage 8: best FX trend filter (written before any stage-8 number was computed)

## Data
FRED H.10 daily noon rates via github.com/datasets/exchange-rates (1971-01-04 .. 2026-09-18), close only.
Universe: 9 G10 currencies vs USD — AUD, CAD, EUR (DKK used as the EUR proxy before 1999-01-04, it was pegged
to the DEM), JPY, NZD, NOK, SEK, CHF, GBP. All converted to "USD per foreign unit" (long = long the currency).
Spot only: carry / swap is NOT included (no rate data) — stated as a caveat.

## Split
Development (choose the filter): 1974-01-01 .. 2004-12-31. Test (once): 2005-01-01 .. 2026-09-18.

## Candidate filters (long/short per currency, position = signal / 60d vol, portfolio = equal risk)
1. TSMOM: sign of the L-day return, L in {21, 63, 126, 252}.
2. EMA crossover sign, (fast, slow) in {(8,32), (16,64), (32,128), (64,256)}.
3. Donchian breakout (+1 on N-day high, -1 on N-day low, hold otherwise), N in {20, 55, 100, 250}.
4. Continuous trend (Baz et al. 2015 style): mean of the three normalised EMA-crossover signals
   (8/24, 16/48, 32/96) passed through x*exp(-x^2/4)/0.89, clipped to [-1, 1].
5. Ensemble: equal-weight average of all 12 rule signals above (no fitting).
That is 14 candidates. Nothing else is tried.

## Costs
0.03% per unit of turnover (spread + slippage, conservative for daily majors). Robustness: x3 cost, 1-day delay.

## Selection rule
The candidate with the highest DEV-period portfolio Sharpe (net of costs) is THE filter. Frozen.

## The market / benchmark
Buy & hold: equal-risk long of the 9 currencies vs USD (i.e. short the dollar), same vol scaling.
Also reported: raw equal-weight buy & hold.

## Primary endpoint
Test-period Sharpe of the selected filter's portfolio > buy & hold Sharpe, with the 90% moving-block
bootstrap CI (block 63 days, 2000 reps) of the Sharpe difference excluding 0. Also the filter's own
Sharpe > 0 with its CI excluding 0, plus the Deflated-Sharpe-style haircut for 14 trials.

## Secondary (reported whatever they show)
All 14 candidates on the test period (so the selection can be judged), per-currency results, sub-periods
2005-2012 / 2013-2019 / 2020-2026, robustness (x3 cost, 1-day delay), and the Oanda hourly pairs 2005-2020
converted to daily as a second data source for the selected filter.

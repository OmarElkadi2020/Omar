# Pre-registration — Stage 6: does cross-sectional structure (PCA absorption, correlation, breadth, dispersion) add crash / end-of-bull information beyond price?

Written and committed BEFORE any feature below was computed or looked at. Nothing is tuned; windows come from the literature.

## Universe / data
- `sp_prices.parquet` (Yahoo, US large caps, 916 tickers 1995-2026, 118 delisted names included -> partial survivorship bias, reported as a caveat).
- Daily returns from `adj_close`, winsorised at +-40%. A stock enters a window only if it has complete returns in that window.
- Market = equal-weight index of all stocks with a return that day.

## Target (primary)
`crash63` = 1 if the market's maximum drawdown over the NEXT 63 trading days (min of I[t+1..t+63]/I[t] - 1) is <= -10%.

## Feature sets (all computed from data up to and including day t)
- **A — price only (market index)**: drawdown from 252d high; returns 21d, 63d, 252d; price / 200d SMA - 1; log realised vol 21d; vol ratio 21d/252d.
- **B = A + cross-section**:
  1. AR = PC1 variance share of the 252d correlation matrix (Kritzman et al. 2011 use top-N/5 eigenvectors; with ~500 stocks and 252 days that is rank-deficient, so PC1 share is used).
  2. dAR = (mean AR last 15d - mean AR last 252d) / std AR last 252d (Kritzman's standardised shift).
  3. average pairwise correlation 63d; 4. its change vs 252d average.
  5. % stocks above own 200d SMA; 6. % above 50d SMA; 7. 21d change of (5).
  8. (52w new highs - 52w new lows) / N.
  9. dispersion = cross-sectional std of 21d returns.
  10. % stocks >= 20% below own 252d high.

## Model / split
- Logistic regression, L2, C = 1.0, standardisation fitted on the development period only. No hyper-parameter search.
- Fit: 1996-01-01 .. 2009-12-31 (labels whose 63d window ends before 2010 only - purge). Test once: 2010-01-01 .. last day with a full 63d forward window.

## Primary endpoint
AUC(B) - AUC(A) on the test period. Success = difference > 0 AND 90% moving-block bootstrap CI (block 126 days, 2000 reps) excludes 0.

## Secondary (reported whatever they show)
1. Brier skill score of A and B vs the test base rate.
2. Practical "exit" test matched on time out of market: go OFF when p > the dev-period 80th percentile of that model's p. Report % of test time OFF, share of the market's >=10% declines covered, and "false OFF" episodes (OFF runs not followed by a >=10% drawdown within 63 days).
3. Event table: every first touch of -5% from the 252d high (dip events). For each: B and A probability at the trigger, and the final depth before a new high. Descriptive only (few events).
4. Univariate test-period AUCs of each cross-sectional feature — descriptive, flagged as multiple comparisons, not used for any conclusion.

## Honesty rules
No change to target, features, windows, model or split after seeing any result. If a bug is found, it is fixed, disclosed in the report, and the run repeated in full.

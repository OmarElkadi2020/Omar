# Pre-registration — Stage 8b: FX carry / cross-sectional filters (written before any 8b number was computed)

Context (disclosed): stage 8 (14 pure time-series trend filters) failed out of sample 2005-2026
(best dev filter: Sharpe 1.11 dev -> -0.11 test). This is a SECOND round on an overlapping test period, so
the trial count is cumulative (14 + 11 = 25) and is used in the multiple-testing haircut.

## Data
Spot: FRED daily (stage 8). Short rates: Jorda-Schularick-Taylor macrohistory `stir`, ANNUAL, to 2020
(via github.com/unbalancedparentheses/forex-centuries). EUR rate = DEU (DNK before 1999 with the DKK splice).
NZD has no rate -> dropped. Universe: AUD CAD EUR JPY NOK SEK CHF GBP vs USD.
Causality: the rate used during calendar year Y is the value for year Y-1 (annual averages would leak).
Total return of long foreign vs USD per day = spot log return + (i_foreign - i_usd)/252.

## Split
Dev 1974-01-01 .. 2004-12-31; test 2005-01-01 .. 2020-12-31 (rates end 2020).

## Candidates (equal risk per currency, 10% vol target each, costs 0.03% per unit turnover)
1. Carry: long the 3 highest-rate, short the 3 lowest-rate currencies.
2-5. Carry filtered by the currency's own trend: keep a carry leg only while that currency's trend agrees
     (long leg needs up-trend, short leg down-trend), else flat. Trend = TSMOM 63d / TSMOM 252d / EMA 32/128 /
     continuous trend (Baz) sign.
6-9. Cross-sectional momentum: long top 3 / short bottom 3 by L-day return, L in {21, 63, 126, 252}.
10. 50/50 carry + XS momentum 63d.
11. Dollar carry (Lustig-Roussanov-Verdelhan 2014): long all foreign currencies when the average foreign rate
    > US rate, else short all.

## Selection and endpoints
Highest dev Sharpe = THE filter. Primary: test Sharpe > 0 with 90% block-bootstrap CI (63d blocks) excluding 0,
AND > buy & hold (equal-risk long all foreign, carry included). Haircut: expected max Sharpe of 25 noise trials.
Secondary: all 11 on test, sub-periods 2005-2012 / 2013-2020, x3 cost, 1-day delay.

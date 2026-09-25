# Pre-registration — Stage 9: can technical indicators alone give a good stock trend filter?
Written before any stage-9 number was computed.

## Data
`sp_prices.parquet` (Yahoo, 916 US large caps 1995-2026, 118 delisted included -> partial survivorship bias,
affects all candidates and the benchmark alike). Daily `adj_close` (dividends included). Daily returns clipped
at +-40% (data errors). Eligible stock on day t: >= 273 days of history and price > $1.
Only price-derived indicators are used (no fundamentals, no macro). Cash earns 0 (conservative for timing rules).

## Split
Dev 1996-01-01 .. 2010-12-31 (select). Test 2011-01-01 .. 2026-09-22 (once).

## Benchmark ("the market")
Equal-weight buy & hold of all eligible stocks (daily rebalanced EW index of the same universe).

## Candidates (long-only, equal weight among selected stocks, position at close t earns t -> t+1)
Per-stock time-series filters (hold the stock only while the rule is ON):
1. price > SMA200 (daily)            2. price > SMA210, checked at month end (Faber 10-month)
3. EMA50 > EMA200 (daily)            4. 12-1 month return > 0, month end
Market timing (hold the whole EW market or cash):
5. EW index > its SMA200 (daily)     6. breadth: % stocks above own SMA200 > 50% (daily)
Cross-sectional selection (month-end rebalance):
7. momentum 12-1 top decile          8. = 7 but only while rule 5 is ON (else cash)
9. 52-week-high proximity top decile 10. lowest 252d volatility decile
11. momentum 12-1 top quintile AND price > SMA200
12. blend: 1/3 each of 7, 9, 10
Costs: 0.10% per unit of turnover.

## Selection / primary endpoint
Highest DEV Sharpe = the selected filter. Primary: test Sharpe > benchmark Sharpe with the 90% moving-block
bootstrap CI (63-day blocks, 2000 reps) of the difference excluding 0. Also reported: CAGR, max drawdown,
Calmar vs benchmark; haircut = expected max Sharpe of 12 noise trials.
Secondary (reported whatever they show): all 12 on test, sub-periods 2011-2018 / 2019-2026, cost x3,
1-day delay, and a "good trend" check: share of the benchmark's >= 20% drawdowns avoided, and time invested.

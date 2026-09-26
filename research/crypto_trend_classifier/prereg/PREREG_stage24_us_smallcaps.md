# Stage 24 pre-registration: the India long-only procedure on US small caps (Russell-2000-like)

Written and committed after downloading the data and before any feature, model or return is computed.

## Question
Stage 23 showed a long-only α in Indian small/mid caps (5.3 %/yr, t 3.12, survivorship-free) and none in large caps
(India top-100, US S&P 500/400). Does the same procedure find α in **US small caps**?

## Data
Hugging Face `defeatbeta/yahoo-finance-data`, `data/US/stock_prices.parquet`: 12,354 US symbols, daily OHLCV,
1994-11 … 2026-09, split-adjusted (checked on AAPL). 1,762 symbols stop trading before 2026-06, so delisted names are
**partly** included — survivorship is reduced, not removed (Yahoo drops many old delistings). Price return only
(dividends not included; same for every strategy and factor).

## Universe (per day, causal) — Russell-2000-like by liquidity
≥ 252 bars of history, close ≥ $5, and rank 1,001 … 3,000 by 30-day median dollar volume (the largest 1,000 are
excluded as large caps). Delisting: a symbol whose data end before 2026-06-01 gets −30 % on its last day
(sensitivity 0 % / −100 %).

## Procedure — identical to stages 18/23, nothing re-tuned
Same features (stage-16 daily + weekly MTF set + stage-17 MA / 52-week-high / residual momentum / idio-vol),
same residual 21-day target, same LightGBM hyper-parameters (`FROZEN_stage18.json`), 21-day book averaging,
next-open execution, annual expanding walk-forward with 7-day + horizon purge, training on this US small-cap data only.
Costs **20 bp** per unit of one-way turnover (small-cap spreads). Test **2011-01-01 … 2026-09-25**.

## Primary endpoint
Long-only top-20 % book: α > 0 with NW (lag 20) t ≥ 3.0 on MKT, SIZE, UMD, STREV, LOWVOL, UMD_VM, TREND built on the
same universe.

## Secondary
40 bp costs; delisting 0 % / −100 %; survivors-only (symbols trading at the end); liquidity halves (ranks 1,001-2,000
vs 2,001-3,000); per-year α; long-short book; +1 day delay.

# Stage 23 pre-registration: the stage-18 India long-only alpha on SURVIVORSHIP-FREE data

Written and committed before the new data are downloaded or any number is computed.

## Question
Stage 18 (today's listings only) found long-only α 8.6 %/yr, t 5.15, with signs that survivorship may inflate it
(α fades with liquidity and with time). Does the α survive when delisted stocks are included?

## Data
NSE daily bhavcopy files (github tilak999/NSE-Data-bank, `data/sec_bhavdata_full_DDMMYYYY.csv`, mid-2010 → 2026-09):
every security that traded that day, delisted ones included. Series EQ, BE, BZ kept (one row per symbol-day,
EQ preferred); SME and other series dropped.
* Returns: `r_t = CLOSE_t / PREV_CLOSE_t − 1`. NSE adjusts PREV_CLOSE on ex-dates for splits/bonus/rights, so
  this is a corporate-action-adjusted price return (ordinary dividends not included — same for every strategy
  and factor). An adjusted price series is rebuilt by chaining r_t; open/high/low are scaled by the same factor.
* Symbol changes: if symbol A stops and symbol B starts within 5 trading days with B's first PREV_CLOSE within
  0.5 % of A's last CLOSE, B is treated as the continuation of A.
* Delisting: when a held stock stops trading without such a link, its position earns a one-off delisting return of
  **−30 %** (Shumway 1997 average for performance delistings) and is removed. Sensitivity: 0 % and −100 %.

## Procedure (identical to stage 18, nothing re-tuned)
Same universe rule (≥ 252 bars, close ≥ ₹10, top 500 by 30-day median traded value), same features (stage-16 daily
+ weekly set + stage-17 MA / 52-week-high / residual-momentum / idio-vol set), same residual 21-day target, same
LightGBM hyper-parameters (`FROZEN_stage18.json`), same 21-day book averaging, next-open execution, 15 bp costs,
annual expanding walk-forward with 7-day + horizon purge. Training uses only this survivorship-free data.
Because the data start mid-2010 and features need ~400 bars of history, **the test period is 2014-01-01 …
2026-09** (first model trained on the survivorship-free rows available before 2014).

## Primary endpoint
Long-only top-20 % book: α > 0 with NW (lag 20) t ≥ 3.0 on MKT, SIZE, UMD, STREV, LOWVOL, UMD_VM, TREND built on
the same survivorship-free universe, 2014-01 … 2026-09.

## Secondary
Delisting-return sensitivity (0 %, −100 %); number of held stocks that got delisted; the same model evaluated
on the survivor-only subset of this data (stocks still trading at the end) to measure the bias directly;
top-300 / top-200 / top-100 liquidity subsets; per-year α; long-short book.

# Stage 18 pre-registration: the stage-17 stock procedure on a never-touched market (India, NSE)

Written and committed after downloading the data and before any feature, model or return is computed.

## Why
US large/mid caps (stages 16, 17) show no technical alpha beyond known factors, and that window is now
closed. The honest way to ask "does the model work on many stocks?" is a **fresh universe** that no
stage of this project has touched: all NSE-listed stocks (github aswinv90/indian-historical-stock-price,
Yahoo-style daily OHLCV + Adj Close, ~2,577 symbols, 1996 → 2026-09).

## Procedure = stage 17, unchanged except where India forces it
* Panel, features (stage-16 daily + weekly MTF set + stage-17 MA / 52-week-high / residual-momentum /
  idio-vol set), residual 21-day target, LightGBM, one refit per year, 7-day + horizon purge: **as stage 17**.
  (beta_60 / resmom_30 are fixed this time; they were all-NaN in stages 16-17.)
* Universe per day: ≥ 252 bars of history, close ≥ ₹10, top 500 by 30-day median traded value.
* Execution next open (split/dividend adjusted), book = average of the last 21 daily books.
* Costs **15 bp** per unit one-way turnover (STT 0.1 % on delivery + charges + slippage).
* Dev: train < 2006-01-01, validate 2006–2010, Optuna 20 trials seed 0, objective = NW t of the primary
  book's α on validation. Test **2011-01-01 … 2026-09-24**.

## Primary book = LONG-ONLY top quintile (unscaled)
Shorting cash equities in India is not practical for most investors, so the primary claim is about what
can actually be held.

## Alpha definition
Daily net long-only returns on MKT (EW universe), SIZE, UMD (12-1), STREV (1-month), LOWVOL (60-day),
UMD_VM and TREND built on the same universe (gross), Newey–West lag 20.

## Primary endpoint
α > 0 with NW t ≥ 3.0, 2011-01 … 2026-09.

## Secondary
Long-short (unscaled and vol-managed), 25 bp costs, top-200 subset, per-year α, +1 day delay,
long-only CAGR / Sharpe / max DD vs the EW market and vs NIFTY-like top-50 EW.

## Caveat written in advance
The dataset has today's listings only (survivorship). The benchmark factors are built on the same
survivor universe, which neutralises most, not all, of the bias in α.

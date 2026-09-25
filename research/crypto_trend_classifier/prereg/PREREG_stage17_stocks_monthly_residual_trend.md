# Stage 17 pre-registration: stocks, monthly residual-trend model (second and last stock attempt)

Written and committed before any stage-17 model is fitted or any stage-17 return is computed.

## Honest context
Stage 16 stocks failed (α −4.7 %/yr, t −1.5, 2011–2026). What we saw of that test: the dev-selected
horizon was 3 days (turnover 0.63/day → ~8 %/yr of costs) and the book loaded on UMD and STREV.
**The 2011–2026 stock window has therefore been looked at once.** This stage re-uses it, so:
* the hurdle is raised from t ≥ 3.0 to **t ≥ 3.5**;
* the controls are made stricter (vol-managed momentum and a trend factor are added);
* there will be no stage 18 on this stock window.

## Design changes (each from published evidence, none from the stage-16 test)
1. **Monthly trend, not days:** H = 21 fixed, book = average of the last 21 daily books (turnover
   ≈ 0.1/day). Refs: Jegadeesh–Titman overlapping portfolios; the user's goal is *trend* prediction.
2. **Residual target:** rank of the forward 21-day return minus β·(EW market forward return),
   β = 252-day rolling beta at t. Ref: Blitz, Huij & Martens (2011) residual momentum.
3. **Extra features** (ranked cross-sectionally like the stage-16 ones): Han–Zhou–Zhu moving-average
   signals log(P/SMA_L) for L ∈ {3,5,10,20,50,100,200,400}; 52-week-high ratio (George–Hwang);
   residual momentum 12-1 and 6-1 scaled by residual vol; idiosyncratic vol 60. Raw context added:
   24-month EW market return and 126-day market vol (Daniel–Moskowitz "panic state").
4. **Volatility-managed book (primary):** net daily L/S return × min(2, 10 % / σ̂), σ̂ = realised vol
   of the unscaled book over the previous 126 days (strictly past). Ref: Barroso & Santa-Clara (2015).

Everything else as stage 16: 861 tickers, trade at next open, 5 bp costs, top/bottom quintile,
LightGBM, expanding walk-forward with one refit per year 2011…2026, training rows purged 7 days
+ horizon before the refit date.

## Tuning (dev only)
Train < 2006-01-01, validate 2006–2010, Optuna TPE 20 trials seed 0, LightGBM params only (H, hold,
target, scaling fixed). Objective = NW t of the validation α (primary book, primary controls).

## Alpha definition
Daily net primary-book returns regressed on: MKT, SIZE, UMD, STREV, LOWVOL (as stage 16) **plus
UMD_VM** (UMD with the same vol management) **plus TREND** (tercile L/S on the mean rank of the eight
MA signals). Newey–West lag 20.

## Primary endpoint
α > 0 with NW t ≥ 3.5 over 2011-01-01 … 2026-09-22.

## Secondary
Unscaled book α; long-only α; costs 10 bp; top-200 and top-50 subsets; per-year α; +1 day delay.

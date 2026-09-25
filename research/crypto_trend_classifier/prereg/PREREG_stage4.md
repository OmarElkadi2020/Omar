# Pre-registration — stage 4: the "buy-the-dip permission filter" (written 2026-09-25T15:42:16Z)
Goal (user): buy corrections inside a known up-trend; stay ON through corrections; switch OFF before a real
reversal; never be as slow as EMA100/200. Nothing below has been computed yet.

## Dip events (independent of every filter, timeframe-native bars)
close crosses below the lower Bollinger band (20, 2 sd); cooldown 12 bars between events.
## Trade per allowed dip
enter at that close; take-profit +3*ATR14, stop -3*ATR14, max 72 bars then exit at close; if a bar touches both,
stop first; cost 0.1% round trip. Result in R = pnl / (3*ATR14) (comparable across assets).
A filter allows the trade only if it is ON at the dip bar's close (causal).

## Primary score = t-stat of allowed trades = mean(R) / std(R) * sqrt(n)
(rewards both quality and number of good dips; buying falling knives lowers it).
## Diagnostics (reported, not optimised) against the ideal labels at a LARGER scale (k=2: corrections smaller
than ~2x cost are ignored; crypto 1h up-trend median ~19 days / ~22%):
- false_offs_per_uptrend: ON->OFF switches inside ideal up-trends, per up-trend
- crash_done_before_off: share of an ideal down-trend's fall already done when the filter first turns OFF (median)
- on_in_bear: share of ideal down-trend bars with the filter ON;  on_in_bull: same for up-trend bars

## Filters
Model: p = frozen universal model C probability. ON when EMA(p, span) > a, OFF when < b (b <= a).
  grid span {1,4,12,24,48} x a {.50,.55,...,.80} x b {.20,.25,...,.50}
Baselines: price > EMA(n) n in {20,50,100,150,200,300}; EMA cross, SuperTrend, online DC (same grids as before);
  plus FIXED EMA200, EMA100 (user request) and "no filter".
## Tuning: ONLY pre-2019 crypto (model probabilities from the 2016/2017/2018 CV folds, trained before each year),
1h + 4h pooled; every filter family gets its best parameters by the primary score; then frozen.
## Test (one run): crypto 2019-2026 (1h, 4h); FX & indices (1h, 4h); 50 stocks (1D).
(These series were already used for the MCC study; the new score and all thresholds come from pre-2019 only.)
## Claim to test: the model filter has a higher primary score than EVERY baseline (tuned and fixed) in each test
cell. Failures are reported as failures.

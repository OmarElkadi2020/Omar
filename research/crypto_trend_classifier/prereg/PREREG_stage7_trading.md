# Pre-registration — Stage 7: our trend indicator as a STANDALONE trend-trading tool
Written before any stage-7 number was computed. No parameter below is tuned; everything is reused frozen.

## Signals (state known at bar close t, position held from close t to close t+1)
- MODEL: frozen universal/walk-forward classifier C, frozen smoothing (span 1, hysteresis 0.204) -> +1/-1.
  crypto: walk-forward probabilities (out_stage2_C.pkl, each year from a model trained only on earlier data).
  FX / indices / 50 stocks: crypto-only universal model (never saw these markets).
- Baselines with parameters tuned on pre-2019 crypto (out_tc_tuning.json, per timeframe; daily stocks use the 4h set):
  EMA cross, Price vs EMA(50), SuperTrend, online directional-change.
- Fixed references: Price vs EMA200, Price vs EMA100, buy & hold.

## Markets / periods
crypto 6 coins listed before 2019 (BTC ETH BNB XRP ADA TRX), 1h and 4h, 2019-01-01 .. 2026-09-23;
6 FX pairs and 9 stock indices, 1h and 4h (Oanda / histdata, ~2005-2020); 50 large US stocks, daily.
Non-crypto: evaluation starts at bar 1000 (indicator warm-up), as in stage 3.

## Trading modes
- LONG-FLAT (primary — the user's use: be in the market only while the indicator says up-trend).
- LONG-SHORT (secondary).
Costs per unit of turnover: crypto 0.10% (fee+slippage), FX 0.01%, indices 0.02%, stocks 0.05%.

## Metrics
Sharpe (annualised), CAGR, max drawdown, Calmar, time in market, round trips per year.

## Primary question
In LONG-FLAT mode, does MODEL have a higher Sharpe than (a) EMA200 and (b) the best-on-average baseline, on the
majority of series in each group (crypto / FX / indices / stocks)? Answered yes/no per group.

## Robustness (reported whatever it shows)
1. One extra bar of execution delay for every signal.
2. Costs doubled.
All numbers reported; nothing changed after seeing results.

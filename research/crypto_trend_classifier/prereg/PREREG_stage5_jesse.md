# Pre-registration — stage 5: Jesse backtest of XAUSTBreakFree with our trend classifier (written 2026-09-25T16:05:11Z)
No backtest of any variant below has been run yet (only a 120-day BTC smoke test of the ORIGINAL strategy).

Engine: Jesse 3.2.2 research.backtest, real 1m candles, futures, leverage 5 (cross), fee 0.02%, start 10,000.
Strategy code: the user's XAUSTBreakFree verbatim, default hyper-parameters (the author's), no re-optimisation.
Variants (identical except regime()):
  ORIGINAL  - 4h SuperTrend(24,4) + EMA10 regime
  MODEL-4h  - regime = our classifier state on the last CLOSED 4h candle (primary: it replaces a 4h regime)
  MODEL-1h  - regime = our classifier state on the current (just closed) 1h candle (secondary)
Classifier: frozen universal model C (FROZEN_v3.json) + frozen smoothing (span 1, hysteresis 0.204) -> +1/-1.
Leak control:
  crypto (10 coins, Binance 1m, 2019-01-01 -> 2026-09-23): walk-forward states (each year from a model trained
    only on data before that year) = out_stage2_C.pkl
  gold XAU/USD, 6 FX pairs, 8 stock indices (Oanda 1m, 2006/2005 -> 2020-05): the crypto-only model (never saw them)
Reported per asset: net profit %, CAGR, Sharpe, max drawdown, trades, win rate; buy & hold for reference.
Question: does MODEL-4h beat ORIGINAL on Sharpe on the majority of assets in each group (crypto / gold+FX / indices)?
All numbers are reported whatever they show.

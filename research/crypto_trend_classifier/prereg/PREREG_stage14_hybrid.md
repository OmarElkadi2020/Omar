# Pre-registration — Stage 14: our model as EARLY ENTRY + the video strategy's ATR exits
Written before any stage-14 number was computed.

## Variants (same simulator, same costs, 1x notional, long AND short like the original)
1. ORIGINAL (approx.): regime = SuperTrend(24,4) + EMA10 on the bar timeframe; enter long when regime up, ADX>=10
   and close > Donchian high (21 bars on 4h ~ 85 1h bars; 14 bars on daily); short mirror (ADX>=15).
2. HYBRID: enter long when OUR MODEL's state flips to up (fresh flip only), short when it flips to down.
3. MODEL-FLIP: always in the market in the model's direction (flip-and-reverse), no stops (reference).
4. BUY & HOLD (reference).
Exits for 1 and 2 are the original strategy's: long stop 3xATR, trailing 6.5xATR from the best close; short stop
4xATR, trailing 3.5xATR, take-profit 5xATR. While in a position new signals are ignored. ATR = ATR(14) of the bar
timeframe; decisions and fills at the bar close.

## Model (fully out-of-sample everywhere)
The stage-12 model (trained on US stocks before 2011, frozen conversion), applied unchanged.
Market-context features: equal-weight basket of the group (coins / FX pairs / indices; gold alone; stocks: all stocks).

## Markets / periods (all unseen by the model)
Crypto 10 coins, 4h and 1D, 2019-01-01 .. 2026-09-23 | gold, 6 FX pairs, 9 stock indices, 4h, 2006 .. 2020-05 |
50 large-cap stocks (set A), daily, 2011 .. 2026. Costs per unit turnover: crypto 0.10%, FX 0.01%,
gold/indices 0.02%, stocks 0.05%.

## Primary question
Per group (crypto 4h, crypto 1D, gold+FX, indices, stocks): does HYBRID have a higher Sharpe than ORIGINAL on the
majority of assets? Two-sided sign test reported. Secondary: CAGR, max drawdown, trades, win rate, vs buy & hold.

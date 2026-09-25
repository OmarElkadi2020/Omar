# Pre-registration — stage 3: stocks & FX (written 2026-09-25T14:47:28Z)
Status when written: stock/FX files downloaded only (row counts printed); NO features, labels, model or baseline
has been computed on any of them.

## Instruments (fixed now)
- FX, Oanda 1h, 2005-01 -> 2020-05: EUR_USD, GBP_USD, AUD_USD, USD_CAD, EUR_JPY, AUD_JPY
- Stock indices, 1h: SPX500, NAS100, US2000, UK100, JP225, AU200, FR40, NL25 (Oanda 2005/2008 -> 2020-05), DAX (histdata 2010-11 -> 2018-12)
- 50 large US stocks, DAILY bars (Johnbrick123/sp500-data, split-adjusted OHLC, 1995 -> 2026-09):
  AAPL MSFT NVDA GOOGL AMZN META BRK-B AVGO TSLA LLY JPM V WMT XOM UNH MA ORCL COST HD PG JNJ NFLX BAC ABBV CRM
  KO CVX MRK AMD PEP TMO ADBE LIN CSCO ACN MCD WFC ABT IBM GE DIS QCOM TXN INTU AMGN CAT PM VZ NOW GS
  (today's largest names -> survivorship bias; it inflates "always long" but not the model-vs-indicator comparison)
- Timeframes: FX & indices 1h and 4h; stocks 1D (a timeframe the model never trained on).

## Target
Same oracle as crypto: Viterbi, 2 states, k = 1, cost = k * trailing sigma_bar * sqrt(24). Evaluate every bar
whose label is final; skip the first 1000 bars of each series (feature warm-up).

## Model
The universal model C from stage 1 (single-asset features only), trained on crypto only, all labels final at
2026-09-24, with its frozen smoothing. No retraining, no tuning, no threshold change on stocks/FX.

## Baselines
EMA cross, Price vs EMA, SuperTrend, Online directional-change with the crypto pre-2019 frozen parameters
(1h params for 1h, 4h params for 4h and 1D). Secondary (reported, not used for pass/fail): each baseline family's
best parameter chosen IN HINDSIGHT on each asset class (an upper bound the model cannot have).

## Pass criteria (per asset class: FX, indices, stocks, and per timeframe)
P1  mean MCC of the model (frozen setting) > mean MCC of every frozen baseline
P2  at <= 2 flips per true trend change, best model MCC > best MCC of every baseline family (frontier)
P3  the model beats the best frozen baseline on >= 60% of the instruments
Anything that fails is reported as a failure. No second attempt on these instruments.

# AGENTS.md: how to get market data in this repo

All research code lives in `research/crypto_trend_classifier/`. **Run every command from that folder**
(`python -m tc.<module>`): the scripts use relative paths such as `data/...` and `prereg/...`.

## Environment
- Python packages: `pip install numba lightgbm pyarrow scikit-learn scipy statsmodels`.
  `optuna` is only needed by the old stage-1…16 tuning code.
- API keys come from environment variables. They are set in the cloud environment, and a new session picks
  them up. **Never write a key into a file, a URL that gets logged, or a commit.**

  | variable | service | used for |
  |---|---|---|
  | `COINALYZE_API_KEY` | api.coinalyze.net (header `api_key`) | perp OHLCV with taker-buy volume, open interest, funding, liquidations, long/short ratio |
  | `BYkaranteli_API_KEY` | bykaranteli.com (header `x-api-key` or `Authorization: Bearer`) | US spot ETF flows, per-issuer ETF data, cross-venue liquidations, DVOL |
  | `COIN_MARKET_CAP_API_KEY` | pro-api.coinmarketcap.com (header `X-CMC_PRO_API_KEY`) | global metrics (BTC dominance, total cap), Fear & Greed (`/v3/fear-and-greed/latest`, `/historical`) |
  | `FRED_API_KEY` | api.stlouisfed.org (query `api_key`) | US net liquidity (WALCL − WTREGEN − RRPONTSYD), DFEDTARU, DGS10 |
  | `FMP_API_KEY` | financialmodelingprep.com | reachable, but ETF endpoints need a paid plan (HTTP 402) |
  | `SEC_API_IO_KEY`, `cryptohftdata_API_KEY` | not used yet | — |

## What is reachable from the cloud container (checked 2026-10)
| source | status |
|---|---|
| `data.binance.vision` (public archive + S3 listing `s3-ap-northeast-1.amazonaws.com/data.binance.vision`) | ✅ spot/futures klines, monthly and daily zip files, incl. delisted pairs |
| `api.binance.com`, `fapi.binance.com` | ❌ HTTP 451 (Binance geo-block). Use the archive or Coinalyze instead |
| api.coinalyze.net, bykaranteli.com, CMC, FRED, `api.alternative.me/fng/` (no key) | ✅ |
| farside.co.uk, sosovalue.com | ❌ 403 / need their own key: use bykaranteli for ETF flows |
| Coinalyze liquidation history | daily from 2022-08 only; 4h ≈ last 8 months; 1h ≈ last 2 months |
| bykaranteli datasets | rolling 12-month window only: always **merge** into the stored file (the downloader does) |

## Stored data (`research/crypto_trend_classifier/data/`) and how to refresh it
| folder / file | content | refresh |
|---|---|---|
| `binance_1h/<COIN>.parquet` | 20 majors, spot 1h OHLCV + quote volume, trades, taker-buy, 2019 → | `python -m tc.download_binance update` |
| `universe_4h/<SYMBOL>.parquet` | all 678 Binance USDT spot pairs incl. delisted, 4h OHLCV, 2017 → | `python -m tc.download_universe update` |
| `coinalyze/{liq_long,liq_short,oi,funding}.parquet` | daily, 10 Binance perps | `python -m tc.download_coinalyze` |
| `coinalyze/lsr_daily.parquet` | daily long/short account ratio BTC, ETH (2020 →) | written by `python -m tc.market_state` |
| `coinalyze/intraday_<date>.parquet` | hourly perp OHLCV+taker-buy, OI, liquidations, funding (BTC ETH SOL LINK) | `python -m tc.intraday_snapshot <YYYY-MM-DD> [days]` |
| `coinalyze/btc_eth_positioning_daily_*.parquet` | snapshot of daily OI/funding/liquidations/L-S for BTC, ETH | one-off snapshot |
| `bykaranteli/etf-flows.csv` | daily net ETF flow, net assets, value traded: BTC, ETH, SOL | `python -m tc.download_bykaranteli` |
| `bykaranteli/etf-issuer-daily.csv` | per ETF: shares outstanding, NAV, implied flow | same |
| `bykaranteli/liquidations-daily.csv` | liquidations per symbol per exchange (all venues) | same |
| `bykaranteli/dvol-daily.csv` | Deribit DVOL (BTC, ETH) | same |

`trend_compass.html` also embeds 4h OHLC (no volume) for BTC ETH BNB XRP SOL, 2019 → 2026-09 (used by stage 29).

## Reports and the scripts that rebuild them
| question | command | report |
|---|---|---|
| Market state: trend on 4h/8h/12h/1D, breadth, waves, pullbacks, positioning, L/S study | `python -m tc.market_state` | `analysis/medium_term/macro/<date>_market_state.md` + `analysis/medium_term/positioning_flows/<date>_positioning_long_short.md` |
| Hour-by-hour anatomy of a sharp move (aggressor side, OI, liquidations, hour-of-day liquidity) | `python -m tc.intraday_snapshot <date> 40` | `analysis/short_term/positioning_flows/<date>_<event>.md` |
| Long-term cycle, weekly swings, long MAs, ETF flows | `python -m tc.longterm_state` | `analysis/long_term/macro/<date>_cycle_and_structure.md` + `analysis/long_term/positioning_flows/<date>_etf_flows.md` |
| Macro context (FRED net liquidity, rates, CMC dominance) | `python -m tc.market_context` | `analysis/medium_term/macro/<date>_market_context.json` (written by the script) |
| Cross-sectional coin ranking (stage 16 rebuilt) | `python -m tc.live16 check` / `rank` | `results_live16_*.csv` |
| Hourly risk watch: alert levels 0 OK / 1 WARNING / 2 DANGER / 3 EXIT (rules in the docstring) | `python -m tc.watch` | `data/watch_log.jsonl` (gitignored) |
| Forward test of frozen rules (stage 31) | `python -m tc.stage31 update` then `run` | `results_stage31_forward.*` |

## Where market analyses go (`research/crypto_trend_classifier/analysis/`)
- `short_term/` (hours to 2 weeks), `medium_term/` (2 weeks to 6 months) and `long_term/` (6 months to 1 year).
- Inside each: `macro/` (whole market: trend, breadth, cycle, macro context), `positioning_flows/` (OI, funding,
  liquidations, L/S, aggressor flow, ETF flows), and one folder per coin named by ticker (`LINK/`, `BTC/`, …).
- Every folder has `README.md` (what belongs there) and `INDEX.md` (table of files). When you add an analysis, add a
  row to the folder's `INDEX.md`, the horizon's `INDEX.md` and `analysis/INDEX.md`. File name: `YYYY-MM-DD_<topic>.md`.
- Short events (hours/days, e.g. intraday anatomy) go under `short_term/`. Research stages stay outside `analysis/`.

## Conventions (keep them)
- Timestamps are UTC, and candles are indexed by **open** time. A daily value of day d is only usable from d+1.
- New hypotheses: write `prereg/PREREG_stageNN_*.md` and commit it **before** any result. Tune on dev, freeze
  in `prereg/FROZEN_stageNN.json`, run the test once, and document both what worked and what did not in
  `README.md` and `TREND_REPORT.md`.
- Judge strategies in money (CAGR, Sharpe, max DD, capture, bootstrap p; see `tc/stage29.py`), not MCC/AUC.
- Market reports are descriptive. Say what the data cannot show (who traded, single-venue data, sample size).
- SuperTrend settings in use: 4h (48,5), 8h (30,4), 12h (20,4), 1D (48,6); vol sizing = `tc/stage30b.vol_lev`.

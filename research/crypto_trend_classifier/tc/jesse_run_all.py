"""Stage 5 runner: ORIGINAL vs MODEL-4h vs MODEL-1h in Jesse on real 1m candles (see PREREG_stage5_jesse.md)."""
import glob
import os
import pickle
import sys
import numpy as np
import pandas as pd
from jesse.research import backtest
import strategies.XAUSTModel as XM
from strategies.XAUSTBreakFree import XAUSTBreakFree
from strategies.XAUSTModel import XAUSTModel

SP = '/tmp/claude-0/-home-user-Omar/82e6854f-0be2-5081-9d89-296fc06d6b3d/scratchpad'
E = 'Binance Perpetual Futures'
CFG = {'starting_balance': 10000, 'fee': 0.0002, 'type': 'futures', 'futures_leverage': 5,
       'futures_leverage_mode': 'cross', 'exchange': E, 'warm_up_candles': 0}
CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'TRXUSDT', 'DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']
OANDA = ['XAU_USD', 'EUR_USD', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'EUR_JPY', 'AUD_JPY', 'SPX500_USD', 'NAS100_USD',
         'US2000_USD', 'UK100_GBP', 'JP225_USD', 'AU200_AUD', 'FR40_EUR', 'NL25_EUR']
STATES = pickle.load(open(f'{SP}/jesse_states.pkl', 'rb'))


def minute_array(name):
    if name in CRYPTO:
        fs = sorted(glob.glob(f'{SP}/m1raw/{name}-1m-*.parquet'))
        d = pd.concat([pd.read_parquet(f) for f in fs])
        d.index = pd.to_datetime(d.timestamp, utc=True)
        d = d[['open', 'high', 'low', 'close', 'volume']]
    else:
        d = pd.read_pickle(f'{SP}/m1raw/OANDA_{name}.pkl')[['open', 'high', 'low', 'close', 'volume']]
    d = d[~d.index.duplicated()].sort_index().astype(float)
    full = d.resample('1min').last()
    full['close'] = full.close.ffill()
    for c in ('open', 'high', 'low'):
        full[c] = full[c].fillna(full.close)   # missing minute = flat candle at the last price
    full['volume'] = full.volume.fillna(0)
    full = full.dropna()
    ts = ((full.index - pd.Timestamp(0, tz='UTC')) // pd.Timedelta('1ms')).astype('int64')
    return np.c_[ts, full.open, full.close, full.high, full.low, full.volume].astype(np.float64)


def run(name):
    arr = minute_array(name)
    first = (pd.Timestamp(arr[0, 0], unit='ms', tz='UTC') + pd.Timedelta(days=120)).ceil('D')  # 4h/1h-aligned start
    t0 = max(pd.Timestamp('2019-01-01', tz='UTC'), first) if name in CRYPTO else first
    t1 = pd.Timestamp('2026-09-23', tz='UTC') if name in CRYPTO else pd.Timestamp(arr[-1, 0], unit='ms', tz='UTC').floor('D')
    s = int(np.searchsorted(arr[:, 0], t0.value // 10 ** 6))
    e = int(np.searchsorted(arr[:, 0], t1.value // 10 ** 6))
    w0 = max(0, s - 60 * 24 * 60)
    sym = name.replace('USDT', '-USDT') if name in CRYPTO else name.replace('_', '-')
    key = f'{E}-{sym}'
    candles = {key: {'exchange': E, 'symbol': sym, 'candles': arr[s:e]}}
    warm = {key: {'exchange': E, 'symbol': sym, 'candles': arr[w0:s]}}
    rows = []
    for var, cls, tf in (('ORIGINAL', XAUSTBreakFree, None), ('MODEL-4h', XAUSTModel, '4h'), ('MODEL-1h', XAUSTModel, '1h')):
        if tf:
            fn = f'{SP}/jrun/states_{name}_{tf}.pkl'
            pickle.dump(STATES[name][tf], open(fn, 'wb'))
            os.environ['TC_STATES'] = fn
            XM._STATES = None
        res = backtest(CFG, [{'exchange': E, 'strategy': cls, 'symbol': sym, 'timeframe': '1h'}],
                       [{'exchange': E, 'symbol': sym, 'timeframe': '4h'}], candles, warmup_candles=warm, fast_mode=True)
        m = res['metrics']
        rows.append(dict(asset=name, variant=var, start=str(t0.date()), end=str(t1.date()),
                         net_profit_pct=m.get('net_profit_percentage'), cagr_pct=m.get('annual_return'),
                         sharpe=m.get('sharpe_ratio'), max_dd_pct=m.get('max_drawdown'), trades=m.get('total'),
                         win_rate=m.get('win_rate'), calmar=m.get('calmar_ratio')))
    bh = arr[e - 1, 2] / arr[s, 2] - 1
    for r in rows:
        r['buy_hold_pct'] = bh * 100
    return rows


if __name__ == '__main__':
    modes = {'crypto': CRYPTO, 'oanda': OANDA, 'late': ['DOGEUSDT', 'ZECUSDT', 'BCHUSDT', 'SOLUSDT']}
    names = modes.get(sys.argv[1], [sys.argv[1]])  # a mode, or a single asset name
    out = []
    for n in names:
        try:
            rs = run(n)
            out += rs
            print(pd.DataFrame(rs)[['asset', 'variant', 'net_profit_pct', 'sharpe', 'max_dd_pct', 'trades']].round(2).to_string(index=False, header=False), flush=True)
        except Exception as ex:
            print('FAILED', n, repr(ex), flush=True)
    pd.DataFrame(out).to_pickle(f'{SP}/out_stage5_{sys.argv[1]}.pkl')

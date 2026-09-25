import pandas as pd, numpy as np, time
from xau_bt import *
res={}
for k in ['XAUUSD','BTCUSD','EURUSD','SPX500','NAS100']:
    t=time.time(); P=Prepared(pd.read_pickle(f'data/{k}.pkl'))
    eq,tr=backtest(P); m=metrics(eq,tr); bh=metrics(buyhold(P),pd.DataFrame(columns=['pnl']))
    print(f"{k}: strat total {m['total']*100:.0f}% CAGR {m['cagr']*100:.1f}% Sharpe {m['sharpe']:.2f} DD {m['maxdd']*100:.0f}% trades {m['trades']} win {m['winrate']*100:.0f}% | B&H CAGR {bh['cagr']*100:.1f}% Sharpe {bh['sharpe']:.2f} DD {bh['maxdd']*100:.0f}%  ({time.time()-t:.0f}s)")
    d=daily(eq); y=d.resample('YE').last(); yr=(y/ y.shift(1).fillna(10000)-1)*100
    b=daily(buyhold(P)).resample('YE').last(); br=(b/b.shift(1).fillna(10000)-1)*100
    print('   year: '+' '.join(f"{i.year}:{a:+.0f}/{c:+.0f}" for i,a,c in zip(yr.index,yr.values,br.values)))
    eq.to_pickle(f'out_eq_{k}.pkl'); tr.to_pickle(f'out_tr_{k}.pkl')

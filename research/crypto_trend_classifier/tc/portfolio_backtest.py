"""Backtest the Binance account as it stood on 2025-11-01 (quantities from the account statement, no trades in the
period) against simple alternatives until 2026-10-05: hold, per-coin daily SuperTrend(48,6) or 4h ST(48,5) exit and
re-entry, BTC only, 50/50 BTC/ETH. Fees 0.1% per switch; a signal on day d's close is traded on day d+1.
One bear-market period only: illustrative, not evidence. Results: analysis/portfolio/2026-10-07_portfolio_review.md.
python -m tc.portfolio_backtest"""
import numpy as np, pandas as pd
from .evaluate import supertrend
Q={'LINK':1024.8323,'ETH':2.257614,'VIRTUAL':4240.5727,'XRP':2334.7563,'RENDER':1428.084,'SUI':1347.0636,'NEIRO':8183422.04,'BTC':0.002604,'ERA':503.6211,'FET':9.68919}
AGG=dict(open='first',high='max',low='min',close='last')
S,E='2025-11-01','2026-10-05'
D,F4={},{}
for c in Q:
    d=pd.read_parquet(f'data/universe_4h/{c}USDT.parquet').astype(float)[list(AGG)]; d.index=d.index.tz_localize(None)
    F4[c]=d; D[c]=d.resample('1D').agg(AGG).dropna()
px=pd.DataFrame({c:D[c].close for c in Q}).loc[S:E]
start=(px.iloc[0]*0+pd.Series({c:D[c].close.asof(pd.Timestamp(S)-pd.Timedelta('1D')) for c in Q}))
V0=(start*pd.Series(Q)).sum(); print('start value',round(V0,0),'end actual',round((px.iloc[-1]*pd.Series(Q)).sum(),0))
fee=0.001
def run(weights, sig=None, freq='1D'):
    # weights: dict coin->initial USD fraction; sig: dict coin->position series (1/0) indexed daily (decided on close, applied next day)
    out=[]
    for c,w in weights.items():
        r=D[c].close.pct_change().loc[S:E]
        if sig is None: pos=pd.Series(1.0,index=r.index)
        else: pos=sig[c].shift(1).reindex(r.index).fillna(0)  # yesterday's close decides today
        trades=pos.diff().abs().fillna(pos.iloc[0])
        eq=(1+pos*r-trades*fee).cumprod()*w*V0
        out.append(eq)
    tot=pd.concat(out,axis=1).sum(axis=1)
    dd=(tot/tot.cummax()-1).min()
    return round(tot.iloc[-1]),round(tot.iloc[-1]/V0-1,3),round(dd,3),int(sum((sig[c].loc[S:E].diff().abs().sum() if sig else 0) for c in weights))
def st_daily(c,p=48,m=6.0):
    return pd.Series(np.where(supertrend(D[c],p,m)==1,1.0,0.0),index=D[c].index)
def st_4h(c):
    s=pd.Series(np.where(supertrend(F4[c],48,5.0)==1,1.0,0.0),index=F4[c].index)
    return s.resample('1D').last()   # state at the day's last 4h close (daily check)
w=((start*pd.Series(Q))/V0).to_dict()
R={}
R['A actual hold (same coins)']=run(w)
R['B same coins + daily ST(48,6) exit/re-entry']=run(w,{c:st_daily(c) for c in Q})
R['C same coins + 4h ST(48,5), checked daily']=run(w,{c:st_4h(c) for c in Q})
R['D all in BTC, hold']=run({'BTC':1.0})
R['E all in BTC + daily ST']=run({'BTC':1.0},{'BTC':st_daily('BTC')})
R['F 50/50 BTC/ETH hold']=run({'BTC':.5,'ETH':.5})
R['G 50/50 BTC/ETH + daily ST']=run({'BTC':.5,'ETH':.5},{c:st_daily(c) for c in ('BTC','ETH')})
for k,v in R.items(): print(f'{k:48s} end {v[0]:>7} ret {v[1]:+.1%} maxDD {v[2]:+.1%} switches {v[3]}')
print((px.iloc[-1]/start-1).round(3).sort_values().to_dict())

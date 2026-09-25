import pandas as pd, numpy as np, statsmodels.api as sm, json
from xau_bt import *
out={}
def reg(eq,P):
    s=daily(eq).pct_change(); b=daily(buyhold(P,eq.index[0])).pct_change()
    df=pd.concat([s,b],axis=1).dropna(); df=df[(df.iloc[:,1]!=0)|(df.iloc[:,0]!=0)]
    X=sm.add_constant(df.iloc[:,1]); f=sm.OLS(df.iloc[:,0],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    n=len(df)/ ((df.index[-1]-df.index[0]).days/365.25)
    return dict(alpha_ann=f.params.iloc[0]*n, alpha_t=f.tvalues.iloc[0], beta=f.params.iloc[1], r2=f.rsquared)
def drop_top(eq,tr,k,cap=10000):
    # approximate: remove top-k trades' pnl from equity path (in $), recompute total & sharpe
    tr=tr.sort_values('pnl',ascending=False); d=daily(eq).copy()
    for _,t in tr.head(k).iterrows(): d[d.index>=t.exit_time.floor('D')]-=t.pnl
    r=d.pct_change().dropna(); return dict(total=d.iloc[-1]/cap-1, sharpe=r.mean()/r.std()*np.sqrt(365))
for k in ['XAUUSD','BTCUSD','EURUSD','SPX500','NAS100']:
    P=Prepared(pd.read_pickle(f'data/{k}.pkl'))
    eq,tr=backtest(P)
    o=dict(base=metrics(eq,tr), reg=reg(eq,P))
    o['drop3']=drop_top(eq,tr,3); o['drop10']=drop_top(eq,tr,10)
    o['slip5bp']=metrics(*backtest(P,slip=0.0005))
    if k=='BTCUSD':
        o['funding10']=metrics(*backtest(P,funding_apr=0.10))
        o['24x7']=metrics(*backtest(Prepared(pd.read_pickle('data/BTCUSD.pkl'),'24x7')))
        for a,b in [('2024-01-01','2026-01-01'),('2024-01-01',None)]:
            e,t=backtest(P,start=a,end=b); o[f'{a}->{b}']=metrics(e,t); o[f'{a}->{b}_reg']=reg(e,P)
    if k=='XAUUSD':
        for a,b in [('2011-09-01','2016-01-01'),('2012-01-01','2019-01-01'),('2016-01-01','2022-03-01')]:
            e,t=backtest(P,start=a,end=b); o[f'{a}->{b}']=metrics(e,t); o[f'{a}->{b}_bh']=metrics(buyhold(P,a,b),pd.DataFrame(columns=['pnl'])); o[f'{a}->{b}_reg']=reg(e,P)
    out[k]=o
    print(k, json.dumps(o,default=lambda x: round(float(x),3),indent=None))
json.dump(out,open('out_run2.json','w'),default=float)

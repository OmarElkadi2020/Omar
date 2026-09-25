import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
from xau_bt import *
R=pd.read_pickle('out_run4_returns.pkl'); cfgs=json.load(open('out_run4_cfgs.json'))
P=Prepared(pd.read_pickle('data/BTCUSD.pkl')); bh=daily(buyhold(P,R.index[0])).pct_change().reindex(R.index).fillna(0)
def S(r): return r.mean()/r.std()*np.sqrt(365) if r.std()>0 else np.nan
def stats(r):
    e=(1+r).cumprod(); y=len(r)/365.25
    return dict(sharpe=round(S(r),2),cagr=round((e.iloc[-1]**(1/y)-1)*100,1),dd=round((e/e.cummax()-1).min()*100,1))
IS=R.loc[:'2021-12-31']; OOS=R.loc['2022-01-01':]
sI=IS.apply(S); sO=OOS.apply(S)
print('Spearman IS vs OOS sharpe across 501 configs:',round(spearmanr(sI,sO,nan_policy='omit')[0],3))
q=pd.qcut(sI.rank(method='first'),5,labels=['Q1 worst','Q2','Q3','Q4','Q5 best IS'])
print('median OOS Sharpe by IS quintile:',sO.groupby(q).median().round(2).to_dict())
best=sI.idxmax(); top=sI.nlargest(int(len(sI)*0.1)).index; top10=sI.nlargest(10).index
rows={'Default (his gold params)':OOS[0],'Best IS Sharpe':OOS[best],'Top-10 ensemble (avg returns)':OOS[top10].mean(axis=1),
      'Random config (median of all)':None,'Buy & hold BTC':bh.loc['2022-01-01':]}
print(f"\nIS 2013-06..2021 -> OOS 2022..2026-09")
print('best IS config:',cfgs[best],'IS sharpe',round(sI[best],2))
for k,v in rows.items():
    if v is None: print(f"{k:32s} OOS sharpe {sO.median():.2f}  (IQR {sO.quantile(.25):.2f}..{sO.quantile(.75):.2f})"); continue
    print(f"{k:32s} IS {S((IS[0] if k.startswith('Default') else IS[best] if k.startswith('Best') else IS[top10].mean(axis=1) if k.startswith('Top') else bh.loc[:'2021-12-31'])):.2f} | OOS {stats(v)}")
print('median OOS sharpe of top-10% IS configs:',round(sO[top].median(),2))
ses=pd.Series([c['session'] for c in cfgs]); print('median OOS by session:',sO.groupby(ses.values).median().round(2).to_dict(),'IS:',sI.groupby(ses.values).median().round(2).to_dict())
# anchored walk-forward, yearly re-optimisation
wf_best=[];wf_top=[];yrs=range(2016,2027)
for Y in yrs:
    tr=R.loc[:f'{Y-1}-12-31']; te=R.loc[f'{Y}-01-01':f'{Y}-12-31']; s=tr.apply(S)
    wf_best.append(te[s.idxmax()]); wf_top.append(te[s.nlargest(10).index].mean(axis=1))
    print(Y,'chosen best',s.idxmax(),'test sharpe best',round(S(te[s.idxmax()]),2),'top10',round(S(wf_top[-1]),2),'default',round(S(te[0]),2),'all-median',round(te.apply(S).median(),2),'B&H',round(S(bh.loc[te.index]),2))
wb=pd.concat(wf_best); wt=pd.concat(wf_top); d0=R.loc[wb.index,0]; med=R.loc[wb.index].median(axis=1)
print('\nWalk-forward 2016..2026 stitched OOS:')
for k,v in {'re-opt best':wb,'re-opt top10 ensemble':wt,'default fixed':d0,'equal-weight ALL 501 configs':R.loc[wb.index].mean(axis=1),'buy&hold':bh.loc[wb.index]}.items(): print(f'  {k:30s}',stats(v))

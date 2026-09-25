import pandas as pd, numpy as np, json, time
from xau_bt import *
rng=np.random.default_rng(7)
HPS=[random_hp(rng) for _ in range(200)]
out={}
def sh(eq,tr): return metrics(eq,tr)['sharpe'] if len(tr)>3 else np.nan
for k,wins in {'XAUUSD':[(y,y+2) for y in range(2007,2021,2)], 'BTCUSD':[(y,y+2) for y in range(2014,2025,2)]}.items():
    P=Prepared(pd.read_pickle(f'data/{k}.pkl')); t=time.time()
    full=[]
    for hp in HPS: full.append(sh(*backtest(P,hp)))
    full=np.array(full); dflt=sh(*backtest(P))
    rows=[]
    for a,b in wins:
        IS=np.array([sh(*backtest(P,hp,f'{a}-01-01',f'{b}-01-01')) for hp in HPS])
        if b+2>2026 and k=='XAUUSD': break
        best=int(np.nanargmax(IS))
        oos=sh(*backtest(P,HPS[best],f'{b}-01-01',f'{b+2}-01-01'))
        oos_all=np.array([sh(*backtest(P,hp,f'{b}-01-01',f'{b+2}-01-01')) for hp in HPS])
        rows.append(dict(IS=f'{a}-{b}',best_IS=IS[best],median_IS=np.nanmedian(IS),OOS_of_best=oos,median_OOS_all=np.nanmedian(oos_all),rank_OOS=float((oos_all<oos).mean())))
        print(k,rows[-1],f'{time.time()-t:.0f}s',flush=True)
    out[k]=dict(full_pct_positive=float((full>0).mean()),full_median=float(np.nanmedian(full)),full_p10=float(np.nanpercentile(full,10)),full_p90=float(np.nanpercentile(full,90)),default=dflt,default_pct=float((full<dflt).mean()),wf=rows)
    print(k,{x:y for x,y in out[k].items() if x!='wf'},flush=True)
json.dump(out,open('out_run3.json','w'),default=float)

import pandas as pd, numpy as np, json, time
from xau_bt import *
df=pd.read_pickle('data/BTCUSD.pkl')
PP={'cme':Prepared(df,'cme'),'24x7':Prepared(df,'24x7')}
rng=np.random.default_rng(11)
cfgs=[]
for i in range(500):
    hp=random_hp(rng); hp['session']='cme' if i%2 else '24x7'; cfgs.append(hp)
dflt=dict(DEFAULT_HP,session='cme'); cfgs=[dflt]+cfgs
t=time.time(); R=[]
for i,hp in enumerate(cfgs):
    h={k:v for k,v in hp.items() if k!='session'}
    eq,tr=backtest(PP[hp['session']],h)
    r=daily(eq).pct_change().fillna(0); R.append(r)
    if i%100==0: print(i,f'{time.time()-t:.0f}s',flush=True)
R=pd.concat(R,axis=1); R.columns=range(len(cfgs))
R.to_pickle('out_run4_returns.pkl'); json.dump(cfgs,open('out_run4_cfgs.json','w'))

import numpy as np, pandas as pd
from tc.features import build_bb
df=pd.read_pickle('data/C_ETHUSDT.pkl'); rng=np.random.default_rng(3)
for tf in ['1h','4h']:
    X=build_bb(df,tf); step=pd.Timedelta(tf)
    for T in rng.integers(20000,len(df)-100,4):
        cut=df.index[T]; Xt=build_bb(df[df.index<cut],tf)
        common=Xt.index[(Xt.index+step)<=cut][-5000:]
        a=Xt.loc[common].values.astype(float); b=X.loc[common].values.astype(float)
        m=~(np.isnan(a)|np.isnan(b))
        print(tf,cut,'max abs diff %.2e'%np.max(np.where(m,np.abs(a-b),0)),'nan mismatch',int(np.sum(np.isnan(a)!=np.isnan(b))), X.shape)

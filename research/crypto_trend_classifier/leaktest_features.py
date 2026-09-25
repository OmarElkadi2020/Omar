import time,pandas as pd,numpy as np
from tc.features import build
df=pd.read_pickle('data/C_BTCUSDT.pkl')
t=time.time(); base,X=build(df,'1h'); print('1h',X.shape,f'{time.time()-t:.1f}s')
t=time.time(); base4,X4=build(df,'4h'); print('4h',X4.shape,f'{time.time()-t:.1f}s')
# LEAK TEST: features recomputed on data truncated at a cutoff must equal the full-data features for every bar
# that is complete before the cutoff. Any difference means a feature peeks at later candles.
rng=np.random.default_rng(1)
for tf,XX in [('1h',X),('4h',X4)]:
    step=pd.Timedelta(tf)
    for T in rng.integers(20000,len(df)-100,4):
        cut=df.index[T]; _,Xt=build(df[df.index<cut],tf)
        common=Xt.index[(Xt.index+step)<=cut]; common=common[-5000:]
        a=Xt.loc[common].values.astype(float); b=XX.loc[common].values.astype(float)
        m=~(np.isnan(a)|np.isnan(b)); rel=np.abs(a-b)/(np.abs(b)+1e-3)
        bad_cols=[XX.columns[j] for j in range(a.shape[1]) if np.nanmax(np.where(m[:,j],rel[:,j],0))>1e-4 or np.any(np.isnan(a[:,j])!=np.isnan(b[:,j]))]
        print(tf,cut,'max rel diff %.2e'%np.max(np.where(m,rel,0)),'nan mismatch',int(np.sum(np.isnan(a)!=np.isnan(b))),'cols:',bad_cols[:8])

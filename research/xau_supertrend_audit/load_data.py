import pandas as pd, numpy as np
D='data/'
def oanda(name):
    d=pd.read_csv(D+f'{name}_oanda_h1.csv',index_col=0,parse_dates=True)[['open','high','low','close']]
    d.index=d.index.tz_localize('UTC'); return d
def ejt(sym):
    e=pd.read_csv(D+f'{sym}_h1.csv',index_col=0,parse_dates=True)[['open','high','low','close']]/100000*(1000 if sym=='XAUUSD' else 1)
    # MT4 "NY-close" server time = New York time + 7h
    e.index=(e.index-pd.Timedelta(hours=7)).tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT')
    e=e[e.index.notna()]; e.index=e.index.tz_convert('UTC'); return e[~e.index.duplicated()]
def splice(a,b):
    cut=a.index[-1]; return pd.concat([a,b[b.index>cut]])
out={}
out['XAUUSD']=splice(oanda('XAU_USD'),ejt('XAUUSD'))
out['EURUSD']=splice(oanda('EUR_USD'),ejt('EURUSD'))
out['SPX500']=oanda('SPX500_USD'); out['NAS100']=oanda('NAS100_USD')
m=pd.concat([pd.read_csv(D+'btc_hist.csv.gz'),pd.read_csv(D+'btc_latest.csv')]).drop_duplicates('timestamp')
m.index=pd.to_datetime(m.timestamp,unit='s',utc=True); m=m[m.volume>0]
b=m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
out['BTCUSD']=b[b.index>='2013-06-01']
for k,v in out.items():
    v=v.sort_index(); v.to_pickle(D+k+'.pkl'); print(k,len(v),v.index[0],v.index[-1],round(v.close.iloc[0],4),round(v.close.iloc[-1],4))

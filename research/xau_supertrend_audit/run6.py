import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from trend_eval import *
from xau_bt import supertrend, ema, session_mask
def regime_4hclose(grid,sess):
    anc=resample(grid,'4h'); anc=anc[session_mask(anc.index,sess)]
    H,L,C=anc.high.values,anc.low.values,anc.close.values; st=supertrend(H,L,C,24,4.0); em=ema(C,10)
    r=np.where(np.isnan(st),0,np.where(C>st,1,-1)); r=np.where((r==1)&(C<em),0,r); r=np.where((r==-1)&(C>em),0,r); r=np.where(np.isnan(em),0,r)
    # value known at close of 4h candle -> apply to 1h bars starting at candle end
    s=pd.Series(r,index=anc.index+pd.Timedelta('4h')); return s.reindex(grid.index,method='ffill').fillna(0).values.astype(np.int8)
def st_only_4hclose(grid,sess):
    anc=resample(grid,'4h'); anc=anc[session_mask(anc.index,sess)]
    H,L,C=anc.high.values,anc.low.values,anc.close.values; st=supertrend(H,L,C,24,4.0)
    r=np.where(np.isnan(st),0,np.where(C>st,1,-1)); s=pd.Series(r,index=anc.index+pd.Timedelta('4h')); return s.reindex(grid.index,method='ffill').fillna(0).values.astype(np.int8)
rows=[]
for A,sess,win in [('XAUUSD','cme',('2007-01-01','2020-05-01')),('BTCUSD','24x7',('2014-01-01','2026-09-25')),('EURUSD','cme',('2007-01-01','2022-03-01'))]:
    df=pd.read_pickle(f'data/{A}.pkl'); df.index=df.index.tz_convert('UTC') if df.index.tz else df.index.tz_localize('UTC')
    CL={'REG4h (as in strategy, 1h close)':regime_classifier(df,'4h',sess)[0],'REG4h on 4h close':regime_4hclose(df,sess),'SuperTrend 4h only (4h close)':st_only_4hclose(df,sess),'EMA200 1h':ema_baseline(df,200),'Always long':np.ones(len(df),np.int8)}
    m=(df.index>=win[0])&(df.index<win[1]); c=df.close.values[m]; rn=np.r_[np.diff(np.log(c)),0]
    dly=df[m].close.resample('1D').last().dropna(); sd=np.log(dly).diff().groupby(dly.index.year).std().median()
    for k in [2,4,8]:
        piv,zz=zigzag(c,k*sd)
        for n,p in CL.items():
            p=p[m]; s=basic_scores(p,zz,rn); g=segment_scores(p,c,piv) if not n.startswith('Always') else None
            row=dict(asset=A,k=k,clf=n,hit=s['hit_when_active'],cover=s['coverage'],wrong_way=s['wrong_way'],mcc=s['mcc'],capture=s['capture'])
            if g is not None:
                fe=g.false_exits.fillna(0).sum(); row.update(end_alert_precision=len(g)/(len(g)+fe), exit_lag_med=g.exit_lag_bars.median(), giveback_med=g.giveback_frac.median(), missed_med=g.missed_frac.median(), entry_lag_med=g.entry_lag_bars.median())
            rows.append(row)
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(2).to_string()); R.to_pickle('out_run6.pkl')

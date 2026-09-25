import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from trend_eval import *
SETS={'XAUUSD':dict(sess='cme',win=('2007-01-01','2020-05-01'),grids={'1h':'data/XAUUSD.pkl','30m':'data/XAUUSD_30m.pkl'}),
      'BTCUSD':dict(sess='24x7',win=('2014-01-01','2026-09-25'),grids={'1h':'data/BTCUSD.pkl','30m':'data/BTCUSD_30m.pkl'}),
      'EURUSD':dict(sess='cme',win=('2007-01-01','2022-03-01'),grids={'1h':'data/EURUSD.pkl'})}
K=[1,2,4,8,16]
rows=[]; segs=[]
for A,cfg in SETS.items():
    for g,path in cfg['grids'].items():
        t0=time.time(); df=pd.read_pickle(path)
        if df.index.tz is None: df.index=df.index.tz_localize('UTC')
        C={}
        C['Strategy regime (4h ST+EMA)'],C['Strategy regime + ADX gate']=regime_classifier(df,'4h',cfg['sess'])
        C['Same regime on 1h candles'],_=regime_classifier(df,'1h',cfg['sess'],exclude_last=(g!='1h'))
        if g=='30m': C['Same regime on 30m candles'],_=regime_classifier(df,'30min',cfg['sess'],exclude_last=False)
        C['Baseline: price vs EMA200']=ema_baseline(df,200)
        C['Baseline: always long']=np.ones(len(df),np.int8)
        m=(df.index>=cfg['win'][0])&(df.index<cfg['win'][1]); d=df[m]; c=d.close.values
        rn=np.r_[np.diff(np.log(c)),0]
        dly=d.close.resample('1D').last().dropna(); sd=np.log(dly).diff().groupby(dly.index.year).std().median()
        for k in K:
            th=k*sd; piv,zz=zigzag(c,th); dp=oracle_dp(c,th/2)
            for name,full in C.items():
                p=full[m]
                for lname,lab in [('Oracle-DP',dp),('ZigZag',zz)]:
                    s=basic_scores(p,lab,rn); s.update(asset=A,grid=g,k=k,theta_pct=th*100,labeler=lname,clf=name,
                        oracle_trend_bars=len(c)/max(len(piv)-1,1))
                    if lname=='ZigZag' and name in ('Strategy regime (4h ST+EMA)','Same regime on 1h candles','Same regime on 30m candles','Baseline: price vs EMA200'):
                        h,cp,mc,ww=circular_null(p,lab,rn,100)
                        s.update(null_hit_p95=np.percentile(h,95),null_hit_med=np.median(h),null_mcc_p95=np.percentile(mc,95),null_capture_p95=np.percentile(cp,95),null_wrongway_p5=np.percentile(ww,5))
                    rows.append(s)
                if name.startswith(('Strategy','Same')):
                    sg=segment_scores(p,c,piv); sg['asset'],sg['grid'],sg['k'],sg['clf']=A,g,k,name; segs.append(sg)
        print(A,g,f'{time.time()-t0:.0f}s',flush=True)
pd.DataFrame(rows).to_pickle('out_run5_scores.pkl'); pd.concat(segs).to_pickle('out_run5_segments.pkl')

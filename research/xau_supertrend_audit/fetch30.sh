cd fd
git sparse-checkout set --no-cone "/pyfinancialdata/data/currencies/oanda/XAU_USD/**"
python3 - <<'PY'
import glob,pandas as pd
fs=sorted(glob.glob('pyfinancialdata/data/currencies/oanda/XAU_USD/*/*.csv'))
d=pd.concat([pd.read_csv(f,index_col=0,parse_dates=True) for f in fs]).sort_index(); d=d[~d.index.duplicated()]
h=d.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
h.index=h.index.tz_localize('UTC'); h.to_pickle('../data/XAUUSD_30m.pkl'); print(len(h),h.index[0],h.index[-1])
PY
git sparse-checkout set --no-cone '/README.md'
cd ..; python3 - <<'PY'
import pandas as pd
m=pd.concat([pd.read_csv('data/btc_hist.csv.gz'),pd.read_csv('data/btc_latest.csv')]).drop_duplicates('timestamp')
m.index=pd.to_datetime(m.timestamp,unit='s',utc=True); m=m[m.volume>0]
b=m.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna(); b=b[b.index>='2013-06-01']
b.to_pickle('data/BTCUSD_30m.pkl'); print(len(b))
PY

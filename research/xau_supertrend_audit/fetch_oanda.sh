set -e
cd fd
for I in XAU_USD SPX500_USD NAS100_USD EUR_USD; do
  git sparse-checkout set --no-cone "/pyfinancialdata/data/currencies/oanda/$I/**"
  python3 - "$I" <<'PY'
import sys,glob,pandas as pd
I=sys.argv[1]
fs=sorted(glob.glob(f'pyfinancialdata/data/currencies/oanda/{I}/*/*.csv'))
d=pd.concat([pd.read_csv(f,index_col=0,parse_dates=True) for f in fs]).sort_index()
d=d[~d.index.duplicated()]
h=d.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
h.to_csv(f'../data/{I}_oanda_h1.csv'); print(I,len(h),h.index[0],h.index[-1])
PY
  git sparse-checkout set --no-cone '/README.md'
done

"""Stage 28 pre-check (dev 2019-2023 only): false SuperTrend down-flips and whether choppiness / momentum at the
flip separate them.  A down-flip is FALSE if within the next 30 days the high exceeds the peak of the up-segment
that just ended.  python -m tc.diag28"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .features import _atr, _adx
from . import stage27 as s27

TFS = ['1h', '4h', '8h', '1d']
FZ = json.load(open('FROZEN_stage27.json'))['choices']
DEV_LO, DEV_HI = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2024-01-01', tz='UTC')


def chop(f, n):
    tr = _atr(f, 1)
    rng = f.high.rolling(n).max() - f.low.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def features(f, tf):
    c = f.close
    lc = np.log(c)
    sig = lc.diff().rolling(200, min_periods=50).std()
    a14 = _atr(f, 14)
    X = pd.DataFrame(index=f.index)
    X['chop14'] = chop(f, 14)
    X['chop50'] = chop(f, 50)
    X['rsi14'] = rsi(c)
    for n in (10, 20, 50, 100):
        X[f'roc{n}'] = (lc - lc.shift(n)) / (sig * np.sqrt(n))
    X['adx14'] = _adx(f, 14)[0]
    X['ema200_dist'] = (c - c.ewm(span=200, adjust=False).mean()) / a14
    X['ema50_dist'] = (c - c.ewm(span=50, adjust=False).mean()) / a14
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    X['macd_hist'] = (m - m.ewm(span=9, adjust=False).mean()) / a14
    X['vol_ratio'] = lc.diff().rolling(20).std() / lc.diff().rolling(200, min_periods=50).std()
    return X


def events(c, tf):
    f = s27.bars(c, tf)
    p, m = FZ[tf]['acc_choice']
    st = s27.s26.st_state(f, int(p), float(m)).astype(int)
    X = features(f, tf)
    if tf != '1d':                                # higher-timeframe context: daily SuperTrend(48,6) state
        d = s27.bars(c, '1d')
        sd = s27.s26.st_state(d, 48, 6.0)
        j = np.searchsorted((d.index + pd.Timedelta('1D')).values, (f.index + pd.Timedelta(hours=s27.HOURS[tf])).values, side='right') - 1
        X['daily_st_up'] = np.where(j >= 0, sd[np.maximum(j, 0)], np.nan)
    H = int(30 * 24 / s27.HOURS[tf])
    fl = np.r_[0, np.diff(st)]
    dns, ups = np.flatnonzero(fl == -1), np.flatnonzero(fl == 1)
    hi, lo, cl = f.high.values, f.low.values, f.close.values
    a14 = _atr(f, 14).values
    rows = []
    for i in dns:
        k = np.searchsorted(ups, i) - 1
        if k < 0 or i + H >= len(cl):
            continue
        u = ups[k]
        peak = hi[u:i + 1].max()
        fut_hi = hi[i + 1:i + 1 + H].max()
        fut_lo = lo[i + 1:i + 1 + H].min()
        r = dict(coin=c, tf=tf, t=f.index[i], false=int(fut_hi > peak),
                 drop_after=np.log(fut_lo / cl[i]), gain_after=np.log(fut_hi / cl[i]),
                 depth_from_peak=(peak - cl[i]) / a14[i], seg_len=np.log1p(i - u), seg_gain=np.log(cl[i] / cl[u]))
        r.update({k2: v for k2, v in X.iloc[i].items()})
        rows.append(r)
    return rows


def main():
    rows = []
    for tf in TFS:
        for c in s27.COINS:
            rows += events(c, tf)
        print('events', tf, flush=True)
    E = pd.DataFrame(rows)
    E.to_pickle('cache_stage16/s28_events.pkl')
    D = E[(E.t >= DEV_LO) & (E.t < DEV_HI)]
    feats = [x for x in E.columns if x not in ('coin', 'tf', 't', 'false', 'drop_after', 'gain_after')]
    out = []
    for tf in TFS:
        d = D[D.tf == tf]
        print(f'\n[{tf}] {FZ[tf]["acc_choice"]}  dev down-flips {len(d)}  FALSE share {d.false.mean():.3f}  '
              f'median 30d drop after TRUE {d[d.false == 0].drop_after.median():+.3f}  after FALSE {d[d.false == 1].drop_after.median():+.3f}')
        for x in feats:
            v = d[x].values
            ok = np.isfinite(v)
            if ok.sum() < 100:
                continue
            a = roc_auc_score(d.false.values[ok], v[ok])
            yrs = []
            for y in range(2019, 2024):
                dy = d[(d.t.dt.year == y)]
                vy = dy[x].values
                oky = np.isfinite(vy)
                if oky.sum() > 30 and dy.false.values[oky].std() > 0:
                    yrs.append(roc_auc_score(dy.false.values[oky], vy[oky]))
            same = np.mean(np.sign(np.array(yrs) - 0.5) == np.sign(a - 0.5)) if yrs else np.nan
            out.append(dict(tf=tf, feature=x, auc=a, sep=abs(a - 0.5), same_sign_years=same, min_year=min(yrs), max_year=max(yrs)))
    R = pd.DataFrame(out)
    R.to_csv('results_diag28_dev_auc.csv', index=False)
    pd.set_option('display.width', 200)
    for tf in TFS:
        print(f'\n[{tf}] top features by separation (AUC for FALSE; >0.5 = higher value -> more likely false)')
        print(R[R.tf == tf].sort_values('sep', ascending=False).head(8).round(3).to_string(index=False))


if __name__ == '__main__':
    main()

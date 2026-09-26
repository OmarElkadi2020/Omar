"""Stage 25: all features (price + order flow + derivatives + cross-market) vs trend indicators on accuracy
and speed (PREREG_stage25_all_features_speed_accuracy.md).  python -m tc.stage25 build|leak|tune|test"""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from .stage14 import frame as frame14, market as market14
from .labels import oracle_labels
from .s12lib import conv_h, ewm_score, lag_metrics, conv_c
from .evaluate import supertrend
from .stage16 import CD, TZ

EVAL = 'BTC ETH BNB SOL XRP ADA DOGE LINK AVAX TRX'.split()
EXTRA = 'DOT LTC BCH ATOM NEAR UNI FIL ETC XLM AAVE'.split()
COINS = EVAL + EXTRA
S25 = 'bnc/s25'
OUT = f'{CD}/s25'
FOLDS = (2022, 2023)
T0, T1 = pd.Timestamp('2024-01-01', tz=TZ), pd.Timestamp('2026-09-01', tz=TZ)
EMB = 24
BLOCKS = ('price', 'flow', 'deriv', 'cross')


# ------------------------------------------------------------------ raw data
def spot(c):
    d = pd.read_pickle(f'{S25}/{c}_spot.pkl')
    d.index.name = 'timestamp'          # 't' clashes with a column name inside features.join_by_close
    return d


def loader(name):
    d = spot(name)
    return d[['open', 'high', 'low', 'close', 'volume']].astype(float)


def funding(c, idx):
    fn = f'bnc/fund/{c}USDT.pkl'
    if not os.path.exists(fn):
        return pd.DataFrame(index=idx)
    d = pd.read_pickle(fn)
    t = pd.to_datetime(d.calc_time.astype('int64'), unit='ms', utc=True)
    s = pd.Series(d.last_funding_rate.astype(float).values, index=t).sort_index()
    s = s[~s.index.duplicated()]
    last = s.resample('4h', label='left', closed='left').last().reindex(idx).ffill(limit=6)
    ev = s.resample('4h', label='left', closed='left').sum(min_count=1).reindex(idx).fillna(0)
    F = pd.DataFrame(index=idx)
    F['fund_last'] = last
    F['fund_sum_3d'] = ev.rolling(18, min_periods=6).sum()
    F['fund_z_30d'] = (last - last.rolling(180, min_periods=60).mean()) / last.rolling(180, min_periods=60).std()
    return F


def flow_block(c, idx):
    sp = spot(c).reindex(idx)
    F = pd.DataFrame(index=idx)
    sh = sp.tb_base / sp.volume.replace(0, np.nan)
    for n in (6, 24, 72):
        F[f'tb_share_{n}'] = sh.rolling(n, min_periods=n // 2).mean()
    lt = np.log1p(sp.trades)
    F['trades_z'] = (lt - lt.rolling(180, min_periods=60).mean()) / lt.rolling(180, min_periods=60).std()
    fn = f'{S25}/{c}_perp.pkl'
    if os.path.exists(fn):
        pp = pd.read_pickle(fn).reindex(idx)
        F['perp_tb_share_24'] = (pp.tb_base / pp.volume.replace(0, np.nan)).rolling(24, min_periods=12).mean()
        F['perp_spot_vol'] = np.log((pp.qvol.rolling(24, min_periods=12).mean()) / sp.qvol.rolling(24, min_periods=12).mean())
    return F


def deriv_block(c, idx):
    F = funding(c, idx)
    fn = f'{S25}/{c}_metrics.pkl'
    if os.path.exists(fn):
        m = pd.read_pickle(fn).reindex(idx)
        oi = np.log(m.sum_open_interest_value)
        for n, k in ((6, '1d'), (18, '3d'), (42, '7d')):
            F[f'oi_chg_{k}'] = oi - oi.shift(n)
        pq = f'{S25}/{c}_perp.pkl'
        if os.path.exists(pq):
            qv = pd.read_pickle(pq).reindex(idx).qvol
            F['oi_to_vol'] = oi - np.log(qv.rolling(180, min_periods=60).mean() * 6)
        for col, nm in (('sum_toptrader_long_short_ratio', 'top_ls'), ('count_long_short_ratio', 'glob_ls')):
            v = np.log(m[col])
            F[nm] = v
            F[f'{nm}_chg_1d'] = v - v.shift(6)
        F['taker_ls'] = np.log(m.sum_taker_long_short_vol_ratio).rolling(6, min_periods=3).mean()
    fp = f'{S25}/{c}_prem.pkl'
    if os.path.exists(fp):
        p = pd.read_pickle(fp).reindex(idx).close
        F['basis'] = p
        F['basis_1d'] = p.rolling(6, min_periods=3).mean()
        F['basis_z_30d'] = (p - p.rolling(180, min_periods=60).mean()) / p.rolling(180, min_periods=60).std()
    return F


def daily_ctx(idx):
    """top-100 breadth / dispersion from the stage-20 daily panel; a daily value is usable after its day closes."""
    D = pd.read_pickle(f'{CD}/crypto20_P.pkl')
    C, el = D['C'].astype(float), D['elig']
    e50 = C.ewm(span=50, adjust=False).mean()
    br = ((C > e50) & el).sum(axis=1) / el.sum(axis=1)
    r30 = np.log(C / C.shift(30)).where(el)
    daily = pd.DataFrame({'breadth_ema50': br, 'disp_30': r30.std(axis=1), 'mkt_ret_30': r30.mean(axis=1)})
    avail = (daily.index + pd.Timedelta('1D')).values             # a day's value is known at its close
    close_t = (idx + pd.Timedelta('4h')).values                   # 4h bar close time
    j = np.searchsorted(avail, close_t, side='right') - 1         # last daily value known at the bar close
    out = pd.DataFrame(np.where(j[:, None] >= 0, daily.values[np.maximum(j, 0)], np.nan), index=idx, columns=daily.columns)
    return out


def price_extra(f):
    c = f.close
    lc = np.log(c)
    r = lc.diff()
    s = r.rolling(180, min_periods=60).std()
    F = pd.DataFrame(index=f.index)
    for n, nm in ((300, 'd50'), (1200, 'd200')):
        F[f'ema_dist_{nm}'] = (lc - np.log(c.ewm(span=n, adjust=False).mean())) / (s * np.sqrt(6))
    for n, nm in ((180, '30d'), (540, '90d')):
        F[f'tsmom_{nm}'] = (lc - lc.shift(n)) / (s * np.sqrt(n))
    for n in (120, 330):
        hi, lo = f.high.rolling(n).max(), f.low.rolling(n).min()
        F[f'don_pos_{n}'] = (c - lo) / (hi - lo).replace(0, np.nan)
    return F


def build():
    os.makedirs(OUT, exist_ok=True)
    closes = {c: loader(c).close for c in COINS}
    mk = market14(closes)
    frames, derivs = {}, {}
    for c in COINS:
        f = frame14(loader, c, '4h', mk)
        f = pd.concat([f, price_extra(f)], axis=1)
        frames[c] = f
        derivs[c] = deriv_block(c, f.index)
        print('price', c, f.shape, flush=True)
    eth = frames['ETH'].close
    btc = frames['BTC'].close
    ethbtc = np.log(eth / btc.reindex(eth.index))
    for c in COINS:
        f = frames[c]
        idx = f.index
        fl = flow_block(c, idx).add_prefix('flow_')
        dv = derivs[c].add_prefix('deriv_')
        cr = pd.DataFrame(index=idx)
        for k, v in derivs['BTC'].reindex(idx).items():
            cr[f'cross_btc_{k}'] = v
        eb = ethbtc.reindex(idx)
        cr['cross_ethbtc_ema42'] = eb - eb.ewm(span=42, adjust=False).mean()
        cr['cross_ethbtc_ema180'] = eb - eb.ewm(span=180, adjust=False).mean()
        cr['cross_ethbtc_mom180'] = eb - eb.shift(180)
        cr = pd.concat([cr, daily_ctx(idx).add_prefix('cross_')], axis=1)
        price_cols = [x for x in f.columns if x not in ('open', 'high', 'low', 'close')]
        g = pd.concat([f[['open', 'high', 'low', 'close']], f[price_cols].add_prefix('price_'), fl, dv, cr], axis=1)
        g = g.loc[:, ~g.columns.duplicated()].astype({k: np.float32 for k in g.columns if k not in ('open', 'high', 'low', 'close')})
        g.to_pickle(f'{OUT}/{c}.pkl')
        print('full', c, g.shape, flush=True)


def load(c):
    return pd.read_pickle(f'{OUT}/{c}.pkl')


def feat_cols(f, blocks=BLOCKS):
    return [x for x in f.columns if x.split('_')[0] in blocks]


# ------------------------------------------------------------------ labels, training rows
def train_rows(f, cut):
    past = f[f.index < cut]
    if len(past) < 800:
        return past.iloc[:0], np.array([])
    lab, fin = oracle_labels(past.close.values, 1.0, return_final=True)
    keep = max(fin - EMB, 0)
    return past.iloc[:keep], (lab[:keep] == 1).astype(int)


def truth(f):
    lab = oracle_labels(f.close.values, 1.0)
    return pd.Series((lab == 1).astype(int), index=f.index)


# ------------------------------------------------------------------ indicators (1 = up, 0 = down)
def ema(c, n):
    return c.ewm(span=n, adjust=False).mean()


def don_state(f, n):
    hi = f.high.rolling(n).max().shift(1).values
    lo = f.low.rolling(n).min().shift(1).values
    c = f.close.values
    st = np.zeros(len(c), int)
    cur = 0
    for i in range(len(c)):
        if c[i] > hi[i]:
            cur = 1
        elif c[i] < lo[i]:
            cur = 0
        st[i] = cur
    return st


def indicator_grid():
    g = [('EMA price', dict(n=n)) for n in (20, 50, 100, 200, 400)]
    g += [('EMA cross', dict(a=a, b=b)) for a in (5, 10, 20, 50) for b in (50, 100, 200, 400) if a < b]
    g += [('SuperTrend', dict(p=p, m=m)) for p in (7, 10, 14, 24, 48) for m in (1.5, 2, 3, 4, 5)]
    g += [('Donchian', dict(n=n)) for n in (20, 55, 100)]
    g += [('TSMOM', dict(n=n)) for n in (12, 42, 84, 180)]
    g += [('MACD', dict())]
    g += [('CUSUM', dict(k=k, h=h)) for k in (0.1, 0.25, 0.5, 1.0) for h in (2, 4, 8, 16, 32)]
    return g


def indicator_state(f, fam, p):
    c = f.close
    if fam == 'EMA price':
        return (c > ema(c, p['n'])).astype(int).values
    if fam == 'EMA cross':
        return (ema(c, p['a']) > ema(c, p['b'])).astype(int).values
    if fam == 'SuperTrend':
        return (np.asarray(supertrend(f, p['p'], float(p['m']))) == 1).astype(int)
    if fam == 'Donchian':
        return don_state(f, p['n'])
    if fam == 'TSMOM':
        return (c > c.shift(p['n'])).astype(int).values
    if fam == 'MACD':
        m = ema(c, 12) - ema(c, 26)
        return (m > m.ewm(span=9, adjust=False).mean()).astype(int).values
    if fam == 'CUSUM':
        lc = np.log(c.values)
        r = np.r_[0.0, np.diff(lc)]
        from .labels import ewm_mean
        sig = np.sqrt(ewm_mean(r * r, 20))
        z = np.r_[0.0, r[1:] / np.maximum(sig[:-1], 1e-9)]
        st = conv_c(np.clip(z, -8, 8), p['k'], p['h'])
        return (pd.Series(st).replace(0, np.nan).ffill().fillna(1).values > 0).astype(int)
    raise ValueError(fam)


# ------------------------------------------------------------------ metrics
def metrics(state, tr, lc):
    fpt, mm, md, mg, n = lag_metrics(np.asarray(state, np.int64), np.asarray(tr, np.int64), np.asarray(lc, float))
    return dict(mcc=matthews_corrcoef(tr, state), fpt=fpt, missed=mm, delay=md, giveback=mg, n_seg=n)


def window(f, lo, hi, trim=0):
    w = np.flatnonzero((f.index >= lo) & (f.index < hi))
    if trim:
        w = w[:-trim]
    return w


def summarize(rows):
    R = pd.DataFrame(rows)
    return dict(mcc=float(R.mcc.mean()), fpt_med=float(R.fpt.median()), delay_med=float(R.delay.median()),
                missed_med=float(R.missed.median()))


def objective(s):
    return s['mcc'] - 0.1 * max(0.0, s['fpt_med'] - 2.0)


# ------------------------------------------------------------------ model
def lgbp(p):
    return dict(objective='binary', learning_rate=p['lr'], num_leaves=p['leaves'], min_data_in_leaf=p['min_leaf'],
                feature_fraction=p['ff'], bagging_fraction=p['bf'], bagging_freq=1, lambda_l2=p['l2'], verbose=-1,
                num_threads=4, seed=0, deterministic=True)


def fit(F, cut, p, blocks=BLOCKS):
    Xs, ys = [], []
    cols = feat_cols(F[EVAL[0]], blocks)
    for c in COINS:
        f = F[c]
        t, y = train_rows(f, cut)
        if len(t):
            Xs.append(t.reindex(columns=cols).values.astype(np.float32))
            ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    return lgb.train(lgbp(p), lgb.Dataset(X, y, feature_name=cols), num_boost_round=p['trees']), cols


def model_state(p_up, span, th):
    s = ewm_score(2 * np.asarray(p_up) - 1, span)
    st = conv_h(s, th, -th)
    return (pd.Series(st).replace(0, np.nan).ffill().fillna(1).values > 0).astype(int)


CONV = [(span, th) for span in (1, 3, 6, 12, 24) for th in (0.0, 0.1, 0.2, 0.3, 0.4)]


def tune():
    import optuna
    F = {c: load(c) for c in COINS}
    T = {c: truth(F[c]) for c in EVAL}
    # indicators on dev folds
    ind = []
    for fam, p in indicator_grid():
        rows = []
        for Y in FOLDS:
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ)
            for c in EVAL:
                f = F[c]
                w = window(f, lo, hi)
                if len(w) < 200:
                    continue
                st = indicator_state(f, fam, p)
                rows.append(metrics(st[w], T[c].values[w], np.log(f.close.values[w])))
        s = summarize(rows)
        ind.append(dict(family=fam, params=json.dumps(p), obj=objective(s), **s))
    I = pd.DataFrame(ind).sort_values('obj', ascending=False)
    I.to_csv(f'{OUT}/dev_indicators.csv', index=False)
    print(I.head(10).to_string(), flush=True)
    log = []

    def obj(trial):
        p = dict(leaves=trial.suggest_int('leaves', 7, 63, log=True), lr=trial.suggest_float('lr', 0.01, 0.1, log=True),
                 trees=trial.suggest_int('trees', 100, 600, step=50),
                 min_leaf=trial.suggest_int('min_leaf', 50, 2000, log=True), ff=trial.suggest_float('ff', 0.3, 1.0),
                 bf=trial.suggest_float('bf', 0.3, 1.0), l2=trial.suggest_float('l2', 1e-3, 100, log=True))
        preds = {}
        for Y in FOLDS:
            lo = pd.Timestamp(f'{Y}-01-01', tz=TZ)
            m, cols = fit(F, lo, p)
            for c in EVAL:
                f = F[c]
                w = window(f, lo - pd.Timedelta('30D'), pd.Timestamp(f'{Y + 1}-01-01', tz=TZ))
                preds[(Y, c)] = (w, m.predict(f[cols].iloc[w].values.astype(np.float32)))
        best = (-np.inf, None, None)
        for span, th in CONV:
            rows = []
            for (Y, c), (w, pr) in preds.items():
                f = F[c]
                st = model_state(pr, span, th)
                keep = f.index[w] >= pd.Timestamp(f'{Y}-01-01', tz=TZ)
                ww = w[keep]
                if len(ww) < 200:
                    continue
                rows.append(metrics(st[keep], T[c].values[ww], np.log(f.close.values[ww])))
            s = summarize(rows)
            o = objective(s)
            if o > best[0]:
                best = (o, dict(span=span, th=th), s)
        trial.set_user_attr('conv', best[1])
        log.append(dict(p, obj=best[0], **best[1], **best[2]))
        print(trial.number, round(best[0], 4), best[1], {k: round(v, 3) for k, v in best[2].items()}, flush=True)
        return best[0]

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=30)
    params = dict(st.best_params, **st.best_trial.user_attrs['conv'])
    best_ind = I.iloc[0]
    json.dump(dict(params=params, dev_obj=st.best_value, best_indicator=dict(family=best_ind.family, params=best_ind.params,
                                                                                obj=float(best_ind.obj)),
                   family_best={fam: g.iloc[0].params for fam, g in I.groupby('family', sort=False)},
                   trials=log, frozen_at=str(pd.Timestamp.utcnow())),
              open('FROZEN_stage25.json', 'w'), indent=1, default=str)
    print('BEST', params, st.best_value, 'best indicator', best_ind.family, best_ind.params, best_ind.obj, flush=True)


def test():
    fz = json.load(open('FROZEN_stage25.json'))
    p = fz['params']
    F = {c: load(c) for c in COINS}
    T = {c: truth(F[c]) for c in EVAL}
    variants = {'MODEL (all blocks)': BLOCKS, 'ablation: price only': ('price',), 'ablation: price+flow': ('price', 'flow'),
                'ablation: price+flow+deriv': ('price', 'flow', 'deriv')}
    states = {}
    for name, blocks in variants.items():
        pr = {c: np.full(len(F[c]), np.nan) for c in EVAL}
        for Y in range(T0.year, T1.year + 1):
            lo, hi = pd.Timestamp(f'{Y}-01-01', tz=TZ), min(pd.Timestamp(f'{Y + 1}-01-01', tz=TZ), T1)
            m, cols = fit(F, lo, p, blocks)
            for c in EVAL:
                f = F[c]
                w = window(f, lo - (pd.Timedelta('30D') if Y == T0.year else pd.Timedelta(0)), hi)
                pr[c][w] = m.predict(f[cols].iloc[w].values.astype(np.float32))
        for c in EVAL:
            v = pr[c]
            ok = ~np.isnan(v)
            st = np.zeros(len(v), int)
            st[ok] = model_state(v[ok], p['span'], p['th'])
            states[(name, c)] = st
        print(name, flush=True)
    rows = []
    cands = [(k, None, None) for k in variants] + [(f'{fam} {pp}', fam, json.loads(pp)) for fam, pp in fz['family_best'].items()]
    for name, fam, pp in cands:
        for c in EVAL:
            f = F[c]
            w = window(f, T0, T1, trim=60)
            st = states[(name, c)] if fam is None else indicator_state(f, fam, pp)
            rows.append(dict(strategy=name, coin=c, **metrics(st[w], T[c].values[w], np.log(f.close.values[w]))))
    R = pd.DataFrame(rows)
    R.to_csv('results_stage25_per_coin.csv', index=False)
    S = R.groupby('strategy').agg(mcc_med=('mcc', 'median'), mcc_mean=('mcc', 'mean'), delay_med=('delay', 'median'),
                                  missed_med=('missed', 'median'), fpt_med=('fpt', 'median'), giveback_med=('giveback', 'median'))
    S = S.sort_values('mcc_med', ascending=False)
    S.to_csv('results_stage25_summary.csv')
    pd.set_option('display.width', 220)
    print(S.round(3).to_string())
    bi = fz['best_indicator']
    bname = f"{bi['family']} {bi['params']}"
    M = R[R.strategy == 'MODEL (all blocks)'].set_index('coin')
    B = R[R.strategy == bname].set_index('coin')
    d_mcc = M.mcc.median() - B.mcc.median()
    wins = int((M.mcc > B.mcc).sum())
    p1 = d_mcc >= 0.05 and wins >= 8
    p2 = M.delay.median() <= 0.8 * B.delay.median()
    p3 = M.fpt.median() <= 1.1 * B.fpt.median()
    print(f'BEST indicator (dev): {bname}')
    print(f'accuracy: dMCC {d_mcc:+.3f}, wins {wins}/10 -> {p1}; speed: delay {M.delay.median():.1f} vs {B.delay.median():.1f} '
          f'-> {p2}; false signals: FPT {M.fpt.median():.2f} vs {B.fpt.median():.2f} -> {p3}')
    print('PRIMARY:', bool(p1 and p2 and p3))


if __name__ == '__main__':
    dict(build=build, tune=tune, test=test)[sys.argv[1]]()

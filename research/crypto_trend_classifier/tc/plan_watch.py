"""Portfolio plan watch (descriptive, not advice). Checks the user's plan levels on COMPLETED candles only and prints
one JSON line, appended to data/plan_log.jsonl (gitignored). Needs COINALYZE_API_KEY (latest candles).
Plan (analysis/portfolio/2026-10-07_plan_v2.md):
  per held coin  12h SuperTrend(20,4) close below the line -> stage A (sell half of that coin)
                 12h close back above the line             -> stage A undone (half may be bought back)
                 daily SuperTrend(48,6) close below         -> stage B (sell the rest of that coin)
  BTC            daily close below its daily line           -> cut remaining alts by half
                 weekly close (Monday-open weeks) > 98,000  -> long-term higher high; plan re-entry rules
  book value     36k (keep 3k cash aside) / 40k (sell a third) / 44k (sell another third)
Lines trail the price, so they are recomputed each run; a 'flip' is a state change vs the previous log line.
python -m tc.plan_watch [holdings.json]"""
import json
import os
import sys
import pandas as pd
from .intraday_snapshot import st_line
from .portfolio import history

LOG = 'data/plan_log.jsonl'
HOLD = 'analysis/portfolio/2026-10-07_holdings.json'
AGG = dict(open='first', high='max', low='min', close='last')
TF = {'12h': (20, 4.0), '1D': (48, 6.0)}
BOOK_LEVELS = (36000, 40000, 44000)


def closed(d, rule, now):
    x = d.resample(rule, origin='epoch').agg(AGG).dropna() if rule != '1W' else \
        d.resample('W-MON', label='left', closed='left').agg(AGG).dropna()
    span = pd.Timedelta('7D') if rule == '1W' else pd.Timedelta(rule)
    return x[x.index + span <= now]


def main():
    H = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HOLD))
    coins = [c for c in H if H[c]['qty'] * 0 == 0]
    data = history(coins + ([] if 'BTC' in coins else ['BTC']))
    now = pd.Timestamp.utcnow()
    prev = None
    if os.path.exists(LOG):
        lines = [x for x in open(LOG) if x.strip()]
        prev = json.loads(lines[-1]) if lines else None
    out = dict(time=str(now.floor('min')), coins={}, alerts=[])
    value = 0.0
    for c, (d, px) in data.items():
        d = d[list(AGG)]
        row = dict(price=px)
        for tf, (p, m) in TF.items():
            x = closed(d, tf, now)
            s, l = st_line(x, p, m)
            row[tf] = dict(state='UP' if s == 1 else 'DOWN', line=float('%.6g' % l), dist_pct=round((l / px - 1) * 100, 1),
                           last_close=float(x.close.iloc[-1]), bar=str(x.index[-1]))
            was = prev and prev['coins'].get(c, {}).get(tf, {}).get('state')
            if was and was != row[tf]['state']:
                stage = {'12h': 'A', '1D': 'B'}[tf]
                verb = 'below' if s == -1 else 'back above'
                out['alerts'].append(f'{c} {tf} closed {verb} its line ({l:.6g}) -> stage {stage}'
                                     + (' undone' if s == 1 else ''))
        if c in H:
            row['value'] = H[c]['qty'] * px
            value += row['value']
        out['coins'][c] = row
    w = closed(data['BTC'][0][list(AGG)], '1W', now)
    out['btc_last_weekly_close'] = dict(close=float(w.close.iloc[-1]), week=str(w.index[-1]), above_98k=bool(w.close.iloc[-1] > 98000))
    if out['btc_last_weekly_close']['above_98k'] and not (prev and prev.get('btc_last_weekly_close', {}).get('above_98k')):
        out['alerts'].append('BTC weekly close above 98,000 (January high): long-term higher high')
    if out['coins']['BTC']['1D']['state'] == 'DOWN' and not (prev and prev['coins']['BTC']['1D']['state'] == 'DOWN'):
        out['alerts'].append('BTC daily close below its daily line -> cut remaining alts by half')
    out['book_value'] = round(value)
    hit = [v for v in BOOK_LEVELS if value >= v]
    prev_hit = [v for v in BOOK_LEVELS if prev and prev.get('book_value', 0) >= v]
    for v in set(hit) - set(prev_hit):
        out['alerts'].append(f'portfolio value reached {v:,}')
    out['nearest'] = sorted(((c, tf, r[tf]['dist_pct']) for c, r in out['coins'].items() for tf in TF
                             if r[tf]['state'] == 'UP'), key=lambda t: -t[2])[:4]
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(out, default=str) + '\n')
    print(json.dumps(out, default=str, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

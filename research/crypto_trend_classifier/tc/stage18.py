"""Stage 18: stage-17 procedure on NSE India (see PREREG_stage18_india_fresh_universe.md).
python -m tc.stage18 panel|build|tune|test"""
import sys
import pandas as pd
from . import stage16, stage17 as s17

s17.MARKET, s17.COST, s17.LONG_ONLY_PRIMARY = 'india', 0.0015, True
s17.TAG, s17.FROZEN, s17.OUT = 'india17', 'FROZEN_stage18.json', 'stage18_india'
s17.T1 = pd.Timestamp('2026-09-25', tz='UTC')

if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'panel':
        stage16.build('india')
    else:
        dict(build=s17.build, tune=s17.tune, test=s17.test)[a]()
